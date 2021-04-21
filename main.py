import time
from agents import *
from math_utils import lerp
from graph_utils import *
import datetime
import os
import torch.multiprocessing as mp
from driver_initializer import *
from call_generator import *
from speed_info import SpeedInfo
import dgl
import glob


def train(
        city: City,
        agent: Agent,
        epochs=10,
        time_steps=100,
        write_log=False,
        log_save_folder='./result',
        save_model=False,
        model_save_folder='./model_data',
        verbose=True,
        epsilon_min=0.0,
        **kwargs
):
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
    :param epsilon_min: epsilon_min
    :return:
    '''
    total_start_time = time.time()

    if agent.do_epsilon_exploration:
        city.epsilon = 1
    else:
        city.epsilon = 0

    # TODO: check seed
    city.random_seed = False
    seed = 100

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
            if agent.do_epsilon_exploration:
                city.epsilon = lerp(1, epsilon_min, (epoch * time_steps + i + 1) / (epochs * time_steps))

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
                if agent.debug_file:
                    print("Example Q values", agent.q_values_saved, file=agent.debug_file)
                    agent.debug_file.flush()

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
             load_directory=None,
             export_result=True,
             save_folder='./result',
             export_q_value_image=False,
             original_G =None,
             export_q_value_image_per=10,
             verbose=False,
             epsilon_min=0.0,
             return_dict=None,
             **kwargs):
    '''
    Function for evaluation.
    :param city: road network environment
    :param agent: agent strategy such as random, proportional, GCN_DQN.
    :param epochs: total number of episode
    :param time_steps: total number of time steps for single episode
    :param load_model: path to model data to load.
    :param load_directory: directory to load model
    :param export_result: whether to export test result.
    :param save_folder: path to create result log.
    :param export_q_value_image: Visualize q values at each time step. This is only for real case.
    :param original_G: Original road network graph. This is NOT a line graph conversed one.
    :param export_q_value_image_per: Export q value images per.
    :param epsilon_min : minimum exploration percentage
    :param return_dict : return dictionary
    :return: mean and std of order response rate.
    '''
    total_assigned = []
    total_missed = []
    total_percentages = []
    start_time = time.time()

    city.random_seed = False

    if load_model is not None:
        print("Load", load_model)
        agent.load_model(load_model)
    elif load_directory is not None:
        # automatically find file from load_directory
        target_name = "%s/%s_%s_*" % (load_directory, city.name, agent.name)
        files = glob.glob(target_name)
        print("Found", files[0])
        agent.load_model(files[0])

    agent.set_eval_mode()

    seed = 0
    current_time_info = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    f = None
    if export_result:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        file_name = "%s/%s_%s_%s_result.txt" % (save_folder, city.name, agent.name, current_time_info)
        f = open(file_name, 'w')

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

        # Set exploration
        city.epsilon = epsilon_min
        print("Epsilon set to", epsilon_min)

        for i in range(time_steps):
            seed = seed + 1
            city.seed = seed
            policy = agent.get_policy(observations)

            # Export Q value images
            if export_q_value_image and isinstance(agent, DQNAgent) and (i % export_q_value_image_per) == 0:
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

    if return_dict is not None:
        return_dict[agent.name] = (total_p, mean, std)

    return mean, std


def make_agent_from_params(city, **kwargs):
    model_type = kwargs["model_type"]
    if model_type == 'random':
        return RandomAgent()
    elif model_type == 'proportional':
        return ProportionalAgent(city, **kwargs)
    else:
        return DQNAgent(city, **kwargs)


def make_city_from_params(**kwargs):
    osmnx_g = ox.load_graphml(kwargs["graph_data"])

    speed_info = SpeedInfo(kwargs["speed_info_data"])
    for edge in osmnx_g.edges(data=True):
        u, v, data = edge
        data['u'] = u
        data['v'] = v
        data['speed_info_closest_road_index'] = speed_info.road_names_dict[data['speed_info_closest_road']]
    g = dgl.DGLGraph()
    g.from_networkx(osmnx_g, edge_attrs=['length', 'u', 'v', 'speed_info_closest_road_index'])
    g_line = g.line_graph(shared=True)

    driver_initializer = BootstrapDriverInitializer(kwargs["driver_initializer_data"])
    call_generator = BootstrapCallGenerator(kwargs["call_generator_data"])
    total_driver_number_per_time = TotalDriverCount(kwargs["total_driver_number_per_time_data"])

    city = City(
        G=g_line,
        call_generator=call_generator,
        driver_initializer=driver_initializer,
        total_driver_number_per_time=total_driver_number_per_time,
        speed_info=speed_info,
        **kwargs
    )
    return city


def evaluate_from_params(**kwargs):
    if kwargs["model_type"] == "gat" or kwargs["model_type"] == "gcn":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(kwargs.get("gpu_id", 0))
        with torch.no_grad():
            city = make_city_from_params(**kwargs) #City(**kwargs)
            agent = make_agent_from_params(city, **kwargs)
            evaluate(city, agent, **kwargs)
    else:
        city = make_city_from_params(**kwargs) #City(**kwargs)
        agent = make_agent_from_params(city, **kwargs)
        evaluate(city, agent, **kwargs)


def train_from_params(**kwargs):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(kwargs.get("gpu_id", 0))
    city = make_city_from_params(**kwargs)  # City(**kwargs)
    agent = make_agent_from_params(city, **kwargs)
    train(city, agent, **kwargs)


def evaluate_using_multiprocessing(common_parameters, kwargs_list):
    save_folder = common_parameters["save_folder"]
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    print("Evaluation started")
    processes = []

    manager = mp.Manager()
    return_dict = manager.dict()
    for kwargs in kwargs_list:
        p = mp.Process(target=evaluate_from_params, kwargs={**common_parameters, **kwargs, "return_dict": return_dict})
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    names = list(return_dict.keys())
    names = sorted(names)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    total_output_file = open("%s/total_result_%s.txt" % (common_parameters['save_folder'], current_time), 'w')

    total_output_file.write("%s\t%s\t%s\t%s\n" % ("name", "total_percentage", "mean", "std"))
    for name in names:
        v = return_dict[name]
        total_p, mean, std = v
        total_output_file.write("%s\t%.6f\t%.6f\t%.6f\n" % (name, total_p, mean, std))


def train_using_multiprocessing(common_parameters, kwargs_list):
    if not os.path.exists(common_parameters["log_save_folder"]):
        os.makedirs(common_parameters["log_save_folder"])
    if not os.path.exists(common_parameters["model_save_folder"]):
        os.makedirs(common_parameters["model_save_folder"])

    print("Train started")
    processes = []
    for kwargs in kwargs_list:
        p = mp.Process(target=train_from_params, kwargs={**common_parameters, **kwargs})
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
