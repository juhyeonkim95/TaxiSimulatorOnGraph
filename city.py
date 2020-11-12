import dgl
import torch
from miscellaneous import *
import random


class City:
    def __init__(self,
                 G: dgl.DGLGraph,
                 call_generator,
                 driver_initializer,
                 total_driver_number_per_time=None,
                 speed_info=None,
                 name='simple_city',
                 driver_coefficient=0.5,
                 consider_speed=False,
                 verbose=False,
                 after_action_random=True
                 ):
        '''
        RL environment for road network.
        :param G: line graph of road graph.
        :param call_generator: CallGenerator object
        :param driver_initializer: DriverInitializer object
        :param total_driver_number_per_time: TotalDriverCount object
        :param speed_info: SpeedInfo object
        :param name: name for city
        :param driver_coefficient: percentage of available drivers. this value is multiplied to all driver number related values.
        :param consider_speed: observe speed?
        :param verbose: print debugging message
        :param after_action_random: after action, put driver on random position or not.
        '''

        self.name = name
        self.roads = []
        self.drivers = []
        self.city_time = 0
        self.city_time_unit_in_minute = 1

        self.driver_uuid = 0
        self.G = G

        self.N = G.number_of_nodes()
        pops = []
        self.road_key_dict = {}

        for i in range(self.N):
            road = Road(i, **self.G.nodes[i].data)
            road.reachable_roads = self.G.out_edges(i)[1].tolist()
            pop = road.popularity
            pops.append(pop)
            self.roads.append(road)
            self.road_key_dict[(road.u, road.v)] = road.uuid

        pops.sort(reverse=True)
        self.actionable_drivers = None
        self.non_actionable_drivers = None

        self.epsilon = 0

        self.call_generator = call_generator
        call_generator.initialize(self)

        self.seed = 0
        self.random_seed = True
        self.driver_initializer = driver_initializer
        self.total_driver_number_per_time = total_driver_number_per_time
        self.driver_coefficient = driver_coefficient
        self.consider_speed = consider_speed
        self.speed_info = speed_info
        self.verbose = verbose
        self.after_action_random = after_action_random

    def get_observation(self):
        if self.consider_speed:
            obs = torch.zeros((self.N, 3))
            for i in range(self.N):
                obs[i][0] = len(self.roads[i].drivers)
                obs[i][1] = len(self.roads[i].calls)
                obs[i][2] = self.roads[i].speed / 24
            return obs
        else:
            obs = torch.zeros((self.N, 2))
            for i in range(self.N):
                obs[i][0] = len(self.roads[i].drivers)
                obs[i][1] = len(self.roads[i].calls)
            return obs

    def set_speed(self):
        if self.speed_info is not None:
            self.speed_info.set_speed(self)

    def update_old_calls(self):
        # remove old calls
        for road in self.roads:
            road.calls = [x for x in road.calls if self.city_time < x.wait_end_time]

    def get_road(self, u, v):
        road_id = self.road_key_dict[(u, v)]
        return road_id

    def generate_calls(self, is_initialize=False):
        self.call_generator.generate_call(self, is_initialize=is_initialize, seed=self.seed if not self.random_seed else None)

    def assign_calls(self):
        """
        Randomly assign call to the drivers at the same road.
        :return
        """
        assigned_call_number = 0

        for road in self.roads:
            assignable_call = min(len(road.calls), len(road.drivers))
            for i in range(assignable_call):
                driver = road.drivers[i]
                driver.assign_call(road.calls[i])
                road.calls[i].served_time = self.city_time
            assigned_call_number += assignable_call
            del road.calls[0:assignable_call]
            del road.drivers[0:assignable_call]

        return assigned_call_number

    def update_drivers_status(self):
        '''
        Check whether driver has finished current call.
        :return:
        '''
        for driver in self.drivers:
            if driver.is_online():
                call = driver.current_serving_call
                if call.served_time + call.duration <= self.city_time:
                    driver.current_serving_call = None
                    driver.road_index = call.e
                    driver.last_road_index = driver.road_index
                    driver.road_position = call.ep
                    self.roads[call.e].drivers.append(driver)

    def charge_drive_time(self):
        for driver in self.drivers:
            if not driver.is_online():
                driver.movable_time = self.city_time_unit_in_minute

    def get_actionable_drivers(self):
        '''
        get actionable / non actionable drivers
        :return: list of actionable / non actionable drivers
        '''
        actionable_drivers = []
        actionable_drivers_count = []
        non_actionable_drivers = []
        non_actionable_drivers_count = []
        for road in self.roads:
            driver_count = 0
            non_driver_count = 0
            for driver in road.drivers:
                if driver.movable_time > 0 and not driver.is_online():
                    road = self.roads[driver.road_index]
                    left_distance = road.length - driver.road_position
                    road_speed_in_meter_per_min = road.speed * 1000 / 60
                    time_to_finish = left_distance / road_speed_in_meter_per_min

                    if time_to_finish > driver.movable_time:
                        driver.road_position += driver.movable_time * road_speed_in_meter_per_min
                        non_actionable_drivers.append(driver)
                        non_driver_count += 1
                        driver.movable_time = 0
                    else:
                        driver.movable_time -= time_to_finish
                        actionable_drivers.append(driver)
                        driver_count += 1
            road.actionable_driver_number = driver_count
            actionable_drivers_count.append(driver_count)
            non_actionable_drivers_count.append(non_driver_count)

        if self.verbose:
            print("Actionable driver number :", sum(actionable_drivers_count))
            print("Non-Actionable driver number :", sum(non_actionable_drivers_count))

        return actionable_drivers, non_actionable_drivers

    def apply_policy(self, policy):
        '''
        Apply policy to controllable agents
        :param policy: list of policy for all roads.
        :return:
        '''
        next_position_ratio = 0
        total_counts = 0

        for driver in self.actionable_drivers:
            road = self.roads[driver.road_index]
            neighbors = road.reachable_roads
            if len(neighbors) == 0:
                next_road_index = -1
            elif len(neighbors) > 1:
                # uniformly random (probability of epsilon)
                if policy is None or (self.epsilon > 0 and np.random.binomial(1, self.epsilon) == 1):
                    next_road_index = np.random.choice(neighbors)

                # random from stochastic policy (probability of 1 - epsilon)
                else:
                    if road.actionable_driver_number == 1:
                        next_road_index = neighbors[np.argmax(policy[driver.road_index])]
                    else:
                        next_road_index = np.random.choice(neighbors, p=policy[driver.road_index])
            else:
                next_road_index = neighbors[0]

            if next_road_index == -1:
                self.drivers.remove(driver)
                self.roads[driver.road_index].drivers.remove(driver)
            else:
                self.roads[driver.road_index].drivers.remove(driver)
                self.roads[next_road_index].drivers.append(driver)
                driver.last_road_index = driver.road_index
                driver.road_index = next_road_index

                if self.after_action_random:
                    driver.road_position = np.random.random() * self.roads[next_road_index].length
                else:
                    road_speed_in_meter_per_min = self.roads[next_road_index].speed * 1000.0 / 60.0
                    x = max(0.0, driver.movable_time - 0.3) * road_speed_in_meter_per_min
                    max_x = 0.9 * self.roads[next_road_index].length
                    min_x = 0.1 * self.roads[next_road_index].length
                    x = max(min(x, max_x), min_x)
                    next_position_ratio += (x / (self.roads[next_road_index].length + 0.01))
                    total_counts += 1
                    driver.road_position = x
                driver.movable_time = 0

        if self.verbose and not self.after_action_random:
            print("After movement position ratio average:", next_position_ratio / total_counts)

    def current_total_call_number(self):
        n = 0
        for road in self.roads:
            n += len(road.calls)
        return n

    def current_total_driver_number(self):
        online = 0
        available = 0
        for driver in self.drivers:
            if driver.is_online():
                online += 1
            else:
                available += 1
        return online + available, online, available

    def onoff_drivers(self):
        '''
        Add or remove drivers to fit with total driver number.
        :return:
        '''
        if self.total_driver_number_per_time is not None:
            expected_driver_number = self.total_driver_number_per_time.get_total_driver_number_at(self.city_time)
            expected_driver_number *= self.driver_coefficient
            expected_driver_number = int(expected_driver_number)
            current_driver_number = len(self.drivers)

            number_to_add = expected_driver_number - current_driver_number
            print("Expected: %d, real: %d" % (expected_driver_number, current_driver_number))

            # remove
            if number_to_add < 0:
                to_remove_n = -number_to_add
                random.shuffle(self.drivers)
                to_remove = self.drivers[0:to_remove_n]
                for driver in to_remove:
                    if not driver.is_online():
                        road = self.roads[driver.road_index]
                        road.drivers.remove(driver)
                del self.drivers[0:to_remove_n]

            # add
            elif number_to_add > 0:
                for i in range(number_to_add):
                    road = random.choice(self.roads)
                    driver = Driver(self.get_next_driver_id(), road.uuid, np.random.random() * road.length)
                    road.drivers.append(driver)
                    self.drivers.append(driver)

    def reset(self):
        '''
        Clear all drivers, calls
        :return:
        '''
        self.city_time = 0
        for road_index in range(self.N):
            road = self.roads[road_index]
            road.drivers.clear()
            road.calls.clear()
        self.drivers.clear()

    def get_next_driver_id(self):
        self.driver_uuid += 1
        return self.driver_uuid - 1

    def initialize(self):
        '''
        Generate idle drivers/calls and assign calls
        :return: initial state
        '''
        driver_distribution = self.driver_initializer.get_initial_distribution(self)

        # generate idle drivers
        for road_index in range(self.N):
            number_of_drivers = int(driver_distribution[road_index]) #np.random.choice([0,1,2,3],p=[0.5,0.2,0.2,0.1])
            road = self.roads[road_index]
            for _ in range(number_of_drivers):
                driver = Driver(self.get_next_driver_id(), road_index, np.random.random() * road.length)
                road.drivers.append(driver)
                self.drivers.append(driver)

        # generate driving drivers
        if self.total_driver_number_per_time is not None:
            expected_driver_number = self.total_driver_number_per_time.get_total_driver_number_at(self.city_time)
            expected_driver_number = int(expected_driver_number * self.driver_coefficient)
            current_driver_number = len(self.drivers)
            working_driver_number_at_the_first = int(expected_driver_number - current_driver_number)
            working_driver_number_at_the_first = max(working_driver_number_at_the_first, 0)
            print("Driving drivers at the first", working_driver_number_at_the_first)
            for i in range(working_driver_number_at_the_first):
                driver = Driver(self.get_next_driver_id(), None, None)
                duration = int(np.random.randint(0, 30, 1))
                end_id = int(np.random.randint(self.N))
                end_road = self.roads[end_id]
                ep = np.random.random() * end_road.length
                call = Call(0, end_id, 0, ep, 0, 5, 1, duration)
                call.served_time = 0
                driver.assign_call(call)
                self.drivers.append(driver)

        print("City initialized with total %d drivers" % len(self.drivers))
        self.generate_calls(is_initialize=True)
        self.assign_calls()
        self.set_speed()
        return self.get_observation()

    def step(self, policy):
        '''
        Single update cycle.
        :param policy: list of policy for all roads.
        :return: next state, assigned call number, missed call number
        '''
        self.charge_drive_time()
        total_call_number_before_assign = self.current_total_call_number()
        self.actionable_drivers, self.non_actionable_drivers = self.get_actionable_drivers()
        self.apply_policy(policy)
        assigned_call_number = self.assign_calls()
        self.city_time += 1

        t, a, b = self.current_total_driver_number()
        if self.verbose:
            print(self.city_time)
            print("Total driver %d, online %d, available %d" % (t, a, b))

            print("current total call %d, assigned %d, percentage %.4f percent" % (total_call_number_before_assign,
                                                                           assigned_call_number,
                                                                           assigned_call_number / (total_call_number_before_assign + 1e-9) * 100))

        before = self.current_total_call_number()
        self.update_old_calls()
        after = self.current_total_call_number()
        missed_call_number = before - after

        self.generate_calls()
        self.update_drivers_status()
        self.onoff_drivers()
        self.set_speed()
        next_state = self.get_observation()
        return next_state, assigned_call_number, missed_call_number