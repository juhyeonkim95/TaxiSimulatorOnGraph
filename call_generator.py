from miscellaneous import *
from city import City
import pandas as pd
import random


class CallGenerator:
    def generate_call(self, city: City, is_initialize=False, seed=None):
        pass

    def initialize(self, city):
        pass

    def get_random_call_at(self, city_time):
        pass


class BootstrapCallGenerator(CallGenerator):
    def __init__(self, file_path, time_interval=1, total_time_step=1440, sample_probability=1.0, default_wait_time=10):
        '''
        Generates call from real data.
        :param file_path: Csv file path
        :param time_interval: Time interval for single step
        :param total_time_step: Number of time steps for one episode
        :param sample_probability: Probability to sample the call
        :param default_wait_time: Default wait time. There is no reliable data for call wait time(remaining time).
        We set this value to 10.
        '''
        self.real_order = pd.read_csv(file_path).values
        self.p = sample_probability
        self.day_orders = None
        self.time_interval = time_interval
        self.total_time_step = total_time_step
        self.default_wait_time = default_wait_time

    def initialize(self, city):
        n_all_orders = len(self.real_order)
        p = self.p
        sampled_orders = np.empty((0, 5))

        # First sample all orders for integer times. ex) 2 in 2.4
        while p > 1:
            sampled_orders = np.vstack((sampled_orders, self.real_order))
            p -= 1

        # then sample left orders. ex) 0.4 in 2.4
        index_sampled_orders = np.where(np.random.binomial(1, p, n_all_orders))
        sampled_orders = np.vstack((sampled_orders, self.real_order[index_sampled_orders]))

        # list for day orders
        self.day_orders = [[] for _ in np.arange(self.total_time_step)]

        for iorder in sampled_orders:
            start = eval(iorder[0])
            end = eval(iorder[1])
            start_node = city.get_road(start[0], start[1])
            end_node = city.get_road(end[0], end[1])
            start_time = iorder[2]
            duration = iorder[3]
            price = iorder[4]

            if type(start_time) == np.float_:
                start_time = int(start_time)
            if type(duration) == np.float_:
                duration = int(duration)

            start_time_index = (start_time // self.time_interval)
            self.day_orders[start_time_index].append([start_node, end_node, start_time, duration, price])

    def get_random_call_at(self, city_time):
        return random.choice(self.day_orders[city_time])

    def generate_call(self, city: City, is_initialize=False, seed=None):
        call_at_t = self.day_orders[city.city_time]
        # generate call objects to each road of the city.
        for call in call_at_t:
            start_id, end_id, _, duration, price = call
            road = city.roads[start_id]
            end_road = city.roads[end_id]
            sp = np.random.random() * road.length
            ep = np.random.random() * end_road.length
            wait_time = np.random.poisson(self.default_wait_time) if not is_initialize else np.random.randint(1, self.default_wait_time)
            price = 1 # or you can set real normalized price.
            call = Call(start_id, end_id, sp, ep, city.city_time, city.city_time + wait_time, price, duration)
            road.calls.append(call)


class RandomCallGenerator(CallGenerator):
    def __init__(self, wait_mean=5, duration_mean=12, coeff=1):
        '''
        Generates call randomly by few parameters.
        :param wait_mean: call wait time.
        :param duration_mean: call duration time.
        :param coeff: This value if multiplied to call number.
        '''
        self.wait_mean = wait_mean
        self.duration_mean = duration_mean
        self.coeff = coeff

    def generate_call(self, city: City, is_initialize=False, seed=None):
        wait_mean = self.wait_mean
        duration_mean = self.duration_mean
        if seed is not None:
            np.random.seed(seed)
        for road_index in range(city.N):
            road = city.roads[road_index]
            number_of_calls = np.random.poisson(road.popularity * (wait_mean if is_initialize else 1) * self.coeff)

            for _ in range(number_of_calls):
                end_road_index = np.random.choice(city.N, 1)[0]
                end_road = city.roads[end_road_index]
                sp = np.random.random() * road.length
                ep = np.random.random() * end_road.length
                wait_time = np.random.poisson(wait_mean) if not is_initialize else np.random.randint(1, wait_mean)
                duration_time = np.random.poisson(duration_mean)
                duration_time = max(duration_time, 2)
                price = 1
                call = Call(road_index, end_road_index, sp, ep, city.city_time, city.city_time + wait_time, price, duration_time)
                road.calls.append(call)
