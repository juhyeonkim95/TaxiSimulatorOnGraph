from GCN import GCN,GAT
import torch.nn.functional as F
import torch
from city import City

from math_utils import softmax, softmax_pow
from graph_utils import *


# class for different agent strategy
class Agent:
    def __init__(self, name):
        self.name = name

    def train(self, next_observations):
        pass

    def get_policy(self, observations):
        pass

    def set_eval_mode(self):
        pass

    def save_model(self, save_model_path):
        pass

    def load_model(self, load_model_path):
        pass


class RandomAgent(Agent):
    def __init__(self):
        super().__init__('random')

    def get_policy(self, observations):
        return None


class ProportionalAgent(Agent):
    def __init__(self, city: City, proportional='order', policy_pow=1, strategy=1):
        if strategy == 0:
            t_name = 'proportional_max'
        elif strategy == 1:
            t_name = 'proportional_%.1f' % (policy_pow)
        elif strategy == 2:
            t_name = 'proportional_exp_%.3f' % (policy_pow)
        super().__init__(t_name)
        self.city = city
        self.order_proportional = (proportional=='order')
        self.policy_pow = policy_pow
        self.strategy = strategy

    def get_policy(self, observations):
        policies = [[] for _ in range(self.city.N)]
        for road in self.city.roads:
            policy = np.zeros((len(road.reachable_roads, )))
            for i, road_index in enumerate(road.reachable_roads):
                v = observations[road_index][1]
                if not self.order_proportional:
                    v = max(v-observations[road_index][0], 0)
                policy[i] = v
            if policy.sum() == 0:
                policy.fill(1)
            if self.strategy == 0:
                policy = np.where(policy == np.amax(policy), 1.0, 0.0)
                policy /= policy.sum()
            elif self.strategy == 1:
                policy /= policy.sum()
                if self.policy_pow != 1:
                    policy = softmax_pow(policy, self.policy_pow)
            else:
                policy = softmax(policy, self.policy_pow)

            policies[road.uuid] = policy
        return policies


class DQNAgent(Agent):
    def __init__(self, city: City, model_type='gcn', policy_pow=1.0, strategy=1):
        if strategy == 0:
            t_name = 'dqn_%s_max' % (model_type)
        elif strategy == 1:
            t_name = 'dqn_%s_%.1f' % (model_type, policy_pow)
        elif strategy == 2:
            t_name = 'dqn_%s_exp_%.1f' % (model_type, policy_pow)
        super().__init__(t_name)

        # reverse direction & add self loop
        newG = city.G.reverse()
        for node in newG.nodes():
            newG.add_edge(node, node)
        self.strategy = strategy

        if model_type == 'gcn':
            self.model = GCN(newG,
                             in_feats=3 if city.consider_speed else 2,
                             n_hidden=8,
                             n_classes=1,
                             n_layers=4,
                             activation=F.relu)
        else:
            self.model = GAT(newG,
                             in_dim=3 if city.consider_speed else 2,
                             num_hidden=8,
                             num_classes=1,
                             num_layers=4,
                             activation=F.relu)

        self.optimizer = torch.optim.Adam(self.model.parameters())

        # define model and target model
        self.model.cuda()
        self.model.train()
        self.target_model = copy.deepcopy(self.model)
        self.target_model.cuda()
        self.target_model_update_period = 10

        self.time_step = 0
        self.city = city

        self.observations = None

        # Q_V(s, t)
        self.q_values = None

        # sigma pi(s,t) Q(s,t)
        self.next_target_expected_return_values = torch.zeros((self.city.N,)).cuda()
        # for memoization
        self.next_target_expected_return_values_valid = np.zeros((self.city.N,), dtype=np.int32)
        self.policy_pow = policy_pow

    def save_model(self, save_path):
        print("SAVING")
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

    def set_eval_mode(self):
        self.model.eval()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self, next_observations):
        if self.time_step % self.target_model_update_period == 0:
            self.update_target_model()

        with torch.no_grad():
            # update target for Q_V(s, t)
            target_q_values = torch.zeros(self.city.N, 1).cuda()
            target_q_values_counts = torch.zeros(self.city.N, 1).cuda()

            # Q_V(s, t+1) = f(s_{t+1})
            next_observations = next_observations.cuda()
            next_target_q_values = self.target_model(next_observations)

            # for memoization
            self.next_target_expected_return_values_valid.fill(-1)

            total_agents = self.city.actionable_drivers + self.city.non_actionable_drivers
            for driver in total_agents:
                # got reward this turn
                if driver.current_serving_call is not None:
                    target_q_values[driver.road_index] += driver.current_serving_call.price
                    target_q_values_counts[driver.road_index] += 1
                else:
                    road = self.city.roads[driver.road_index]
                    neighbors = self.city.roads[driver.road_index].reachable_roads

                    # (1) controllable agents
                    if driver.road_position + road.speed * self.city.city_time_unit_in_minute > road.length and len(neighbors) > 1:
                        if self.next_target_expected_return_values_valid[driver.road_index] == -1:
                            # pi(s, t+1)
                            next_target_policy = self.get_policy_from_action_values(next_target_q_values[neighbors].squeeze())

                            # sigma pi(s,t+1) Q(s,t+1)
                            m = torch.dot(next_target_q_values[neighbors].squeeze(), next_target_policy)

                            # set result and memorize it.
                            self.next_target_expected_return_values[driver.road_index] = m
                            self.next_target_expected_return_values_valid[driver.road_index] = 1
                        else:
                            # just return before calculated value.
                            m = self.next_target_expected_return_values[driver.road_index]

                    # (2) non-controllable agents
                    else:
                        m = next_target_q_values[driver.road_index]

                    target_q_values[driver.road_index] += 0.9 * m
                    target_q_values_counts[driver.road_index] += 1

            # For some roads, there are no drivers
            no_info = (target_q_values_counts == 0).int()

            # for road with >= 1 drivers: sum / (N + 0) = avg
            # for road with 0 driver: 0 / (0 + 1) = 0
            target_q_values /= (target_q_values_counts + no_info)

        # for road with 0 driver : don't have to update.
        # but to give a penalty for uncertainty(no experience), multiply by 0.9
        target_q_values += self.q_values * no_info * 0.9

        # should be between (0, 1)
        target_q_values = torch.clamp(target_q_values, min=1e-8, max=1)

        # set loss as weighted MSE
        difference = (self.q_values - target_q_values)
        weighted_mse = (difference ** 2) * (target_q_values_counts + no_info)
        loss = torch.mean(weighted_mse)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # for debugging
        debug = False
        if self.city.city_time % 10 == 0 and debug:
            debug_target_q_values = target_q_values.squeeze().cpu().tolist()
            debug_q_values = self.q_values.squeeze().cpu().tolist()
            index = list(range(self.city.N))
            debug_q_values_info = list(zip(debug_target_q_values, debug_q_values, index))
            debug_q_values_info.sort(reverse=True, key=lambda x:x[0])
            print(debug_q_values_info[0:30])
            print(loss)

    def get_policy(self, observations, use_target_model=False, to_numpy=True):
        policy = [None for _ in range(self.city.N)]
        model = self.model if not use_target_model else self.target_model

        # Q_V(j, t) = f(s_t)
        q_values = model(observations.cuda())

        for v in range(self.city.N):
            out_nodes = self.city.roads[v].reachable_roads
            if len(out_nodes) == 0:
                policy[v] = [-1]
            else:
                possible_action_values = q_values[out_nodes].squeeze()
                policy_v = self.get_policy_from_action_values(possible_action_values)
                if to_numpy:
                    policy_v = policy_v.cpu().detach().numpy()
                policy[v] = policy_v
        self.q_values = q_values
        self.observations = observations
        return policy

    def get_policy_from_action_values(self, q_values: torch.Tensor):
        strategy = self.strategy
        if strategy == 0:
            m = torch.max(q_values)
            p = (q_values == m).float()
        elif strategy == 1:
            if q_values.sum() == 0:
                p = torch.ones_like(q_values)
                p = p / p.sum()
            else:
                p = q_values / q_values.sum()
            p = p / torch.max(p)
            p = p**self.policy_pow
        else:
            # if q_values.sum() == 0:
            #     p = torch.ones_like(q_values)
            #     p = p / p.sum()
            # else:
            #     p = q_values / q_values.sum()
            q_values_max = torch.max(q_values)
            p = torch.exp((q_values-q_values_max)*self.policy_pow)

        if torch.isnan(p).any():
            print(q_values.sum())
            print(q_values)
            print("NAN")
        p /= p.sum()
        return p
