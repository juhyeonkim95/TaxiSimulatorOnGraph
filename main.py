import time
from agents import *
from math_utils import lerp
from graph_utils import *
import datetime
import os


def train(
        city: City,
        agent: Agent,
        epochs=10,
        time_steps=100,
        write_log=False,
        log_save_folder='./result',
        save_model=False,
        model_save_folder='./model_data',
        verbose=True):
    '''
    Function for training.
    :param city: road network environment
    :param agent: agent strategy such as random, proportional, GCN_DQN.
    :param epochs: total number of episode
    :param time_steps: total number of time steps for single episode
    :param write_log: whether to write log
    :param log_save_folder: save log folder
    :param save_model: whether to save model
    :param model_save_folder: save model folder
    :param verbose: print debugging message or not
    :return:
    '''
    total_start_time = time.time()
    city.epsilon = 1

    city.random_seed = True
    seed = 0

    log_file = None
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = "%s_%s_%s"% (city.name, agent.name, current_time)

    if save_model:
        if not os.path.exists(model_save_folder):
            os.makedirs(model_save_folder)

    if write_log:
        if not os.path.exists(log_save_folder):
            os.makedirs(log_save_folder)
        log_file = open("%s/%s_train_timestamp.txt" % (log_save_folder, file_name), 'w')

    for epoch in range(epochs):
        city.reset()
        city.seed = seed

        # Initialize city.
        observations = city.initialize()
        assigned_epoch = 0
        missed_epoch = 0
        start_time = time.time()

        for i in range(time_steps):
            seed = seed + 1
            city.seed = seed
            # get policy from s_t
            policy = agent.get_policy(observations)
            # apply policy to city and get observation/number of assigned order/missed order.
            next_observations, assigned, missed = city.step(policy)
            assigned_epoch += assigned
            missed_epoch += missed

            # training
            agent.train(next_observations)

            observations = next_observations

            # epsilon for exploration.
            city.epsilon = lerp(1, 0.01, (epoch * time_steps + i + 1) / (epochs * time_steps))

            so_far_hit_rate = assigned_epoch / (assigned_epoch + missed_epoch + 1e-8)
            if verbose:
                print("hit rate so far: %.4f" % so_far_hit_rate)

            # write log for every 10 time steps.
            if i % 10 == 0 and write_log:
                end_time = time.time()
                elapses_time = end_time - start_time
                s = time.strftime('%H:%M:%S', time.gmtime(elapses_time))
                log_file.write('%d, %d, %s, %.4f\n' % (epoch, i, s, so_far_hit_rate))
                log_file.flush()

        # train for one episode finished. write log.
        end_time = time.time()
        elapses_time = end_time - start_time
        s = time.strftime('%H:%M:%S', time.gmtime(elapses_time))
        if write_log:
            log_file.write('Total %d, %s, %.4f\n' % (epoch, s, (assigned_epoch / (assigned_epoch + missed_epoch + 1e-8))))

    if log_file is not None:
        log_file.close()

    s = time.strftime('%H:%M:%S', time.gmtime(time.time() - total_start_time))
    print("Total train, ", s)

    # save model data
    if save_model:
        agent.save_model("%s/%s_model_data" % (model_save_folder, file_name))

    return agent


def evaluate(city: City,
             agent: Agent,
             epochs=10,
             time_steps=100,
             load_model=None,
             export_result=True,
             save_folder='./result',
             export_q_values=False,
             original_G =None,
             export_q_values_per=10,
             verbose=True):
    '''
    Function for evaluation.
    :param city: road network environment
    :param agent: agent strategy such as random, proportional, GCN_DQN.
    :param epochs: total number of episode
    :param time_steps: total number of time steps for single episode
    :param load_model: path to model data to load.
    :param export_result: whether to export test result.
    :param save_folder: path to create result log.
    :param export_q_values: Visualize q values at each time step. This is only for real case.
    :param original_G: Original road network graph. This is NOT a line graph conversed one.
    :param export_q_values_per: Export q value images per.
    :return: mean and std of order response rate.
    '''
    total_assigned = []
    total_missed = []
    total_percentages = []
    start_time = time.time()

    city.random_seed = False

    if load_model is not None:
        agent.load_model(load_model)

    agent.set_eval_mode()

    seed = 0
    file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    f = None
    if export_result:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        f = open("%s/%s_%s_%s_result.txt" % (save_folder, city.name, agent.name, file_name), 'w')
        f.write('%s\t%s\t%s\n' % ('total_assigned', 'total_missed', 'served_rate'))
        f.flush()

    for e in range(epochs):
        total_assigned_epoch = 0
        total_missed_epoch = 0

        np.random.seed(e)

        # Initialize city.
        city.reset()
        city.seed = seed
        observations = city.initialize()

        # no exploration
        city.epsilon = 0

        for i in range(time_steps):
            seed = seed + 1
            city.seed = seed
            policy = agent.get_policy(observations)

            # Export Q value images
            if export_q_values and isinstance(agent, DQNAgent) and (i%export_q_values_per) == 0:
                q_values = agent.q_values.cpu().squeeze().tolist()
                for edge in original_G.edges(data=True):
                    u, v, data = edge
                    road_index = city.get_road(u, v)
                    q = q_values[road_index]
                    q = max(min(q, 1), 0)
                    data['q_value'] = q
                    data['q_value_color'] = (q, 1-q, 0)
                ec = [k['q_value_color'] for u, v, k in original_G.edges(data=True)]
                ox.plot_graph(original_G, fig_height=10, show=False, save=True,
                              filename='q_values_at%d' % i, file_format='svg', node_size = 0, edge_color = ec)
                print("exported Graph")

            next_observations, assigned, missed = city.step(policy=policy)
            total_assigned_epoch += assigned
            total_missed_epoch += missed
            so_far_hit_rate = total_assigned_epoch / (total_assigned_epoch + total_missed_epoch + 1e-8)
            if verbose:
                print("hit rate so far: %.4f" % so_far_hit_rate)

            observations = next_observations

        hit_rate = total_assigned_epoch / (total_assigned_epoch + total_missed_epoch)
        print("Order response rate in this episode:", hit_rate)
        total_assigned.append(total_assigned_epoch)
        total_missed.append(total_missed_epoch)
        total_percentages.append(hit_rate)

        # write final order response rate in this episode.
        if export_result:
            f.write('%d\t%d\t%.4f\n' % (total_assigned_epoch, total_missed_epoch, hit_rate))
            f.flush()

    # print elapsed time
    end_time = time.time()
    elapses_time = end_time - start_time
    s = time.strftime('%H:%M:%S', time.gmtime(elapses_time))
    print(s)

    # total order response rate.
    total_missed_n = sum(i for i in total_missed)
    total_assigned_n = sum(i for i in total_assigned)
    total_p = total_assigned_n / (total_assigned_n + total_missed_n)
    print("Total percentage:", total_p)

    # mean, std of order response rate for each episode.
    import statistics
    mean = statistics.mean(total_percentages)
    if len(total_percentages) > 1:
        std = statistics.stdev(total_percentages)
    else:
        std = 0
    print("Total percentage 2:", mean, std)



    ## this is for test.
    # export final result.

    if export_result:
        f.write('%d\t%d\t%.4f\n' % (total_assigned_n, total_missed_n, total_p))
        f.close()

    return mean, std