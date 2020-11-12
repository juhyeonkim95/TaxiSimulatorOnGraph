import pandas as pd
import numpy as np
from math_utils import lerp


class DriverInitializer:
    def get_initial_distribution(self, city):
        pass


class RandomDriverInitializer(DriverInitializer):
    def __init__(self, driver_n=None, driver_p=None):
        '''
        Initializes driver randomly with few parameters.
        :param driver_n: Possible driver numbers on each road.
        :param driver_p: Probability distribution for driver_n.
        '''
        if driver_p is None:
            driver_p = [0.2, 0.8]
        if driver_n is None:
            driver_n = [0, 1]
        self.driver_n = driver_n
        self.driver_p = driver_p

    def get_initial_distribution(self, city):
        initial_distribution = np.random.choice(self.driver_n, city.N, p=self.driver_p)
        return initial_distribution


class BootstrapDriverInitializer(DriverInitializer):
    def __init__(self, data_path):
        '''
        Initializes driver with real data.
        :param data_path: path to initial driver distribution.
        '''
        self.initial_distribution = pd.read_csv(data_path)

    def get_initial_distribution(self, city):
        initial_distrib = np.zeros((city.N,), dtype=np.int32)
        for index, row in self.initial_distribution.iterrows():
            road_id = eval(row['road_id'])
            idle_driver_count = row['idle_driver_count']
            road_index = city.get_road(road_id[0], road_id[1])
            real_count = idle_driver_count * city.driver_coefficient
            v = int(real_count)
            real_count_remainder = real_count - v
            if np.random.rand() < real_count_remainder:
                v = v + 1
            initial_distrib[road_index] = v

        return initial_distrib


class TotalDriverCount:
    def __init__(self, input_file_name, unit_time=10):
        '''
        Initializes total driver number with real data.
        :param input_file_name: path to total driver number for each time step.
        :param unit_time: unit time for total driver number data. Default is 10. We will linearly interpolate data.
        '''
        data = pd.read_csv(input_file_name)
        self.data = {}
        for index, row in data.iterrows():
            self.data[row['time_stamp']] = row['total_driver_count_mean']
        self.unit_time = unit_time

    def get_total_driver_number_at(self, city_time):
        k1 = city_time // self.unit_time
        r = city_time / self.unit_time - k1
        k2 = k1 + 1
        k1 = k1 % len(self.data)
        k2 = k2 % len(self.data)
        v1 = self.data[k1]
        v2 = self.data[k2]
        v = lerp(v1, v2, r)
        return v