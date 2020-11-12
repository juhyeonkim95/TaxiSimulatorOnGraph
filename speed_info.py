import pandas as pd
import numpy as np


class SpeedInfo:
    def __init__(self, file_path, unit_time=60):
        '''
        Speed information for each road.
        :param file_path: Path to speed info data.
        :param unit_time: Default is 60. We will use linear interpolation for intermediate values.
        '''
        data = pd.read_csv(file_path, encoding='euc-kr')
        d = data.drop(columns=['일자', '링크아이디', '거리', '차선수']).groupby('도로명')
        speed_mean = d.agg(['mean'])
        speed_std = d.agg(['std'])
        road_names = speed_mean.index

        speed_mean = speed_mean.values
        speed_std = speed_std.values

        speed_mean[np.isnan(speed_mean)] = 24
        speed_std[np.isnan(speed_std)] = 4

        speed_std = speed_std / 1.5

        self.speed_mean = speed_mean
        self.speed_std = speed_std
        self.road_names = road_names
        self.road_names_dict = {}
        for i, k in enumerate(road_names):
            self.road_names_dict[k] = i
        self.unit_time = unit_time
        self.max_time = 24

    def set_speed(self, city):
        k1 = city.city_time // self.unit_time
        k2 = k1 + 1
        r = (city.city_time / self.unit_time) - k1

        k1 = k1 % self.max_time
        k2 = k2 % self.max_time

        lerped_mean = self.speed_mean[:,k1] * (1 - r) + self.speed_mean[:,k2] * r
        lerped_std = self.speed_std[:,k1] * (1 - r) + self.speed_std[:,k2] * r

        for road in city.roads:
            road_index = road.speed_info_closest_road_index
            f = np.random.normal(lerped_mean[road_index], lerped_std[road_index])
            f = max(f, 5)
            f = min(f, 50)
            road.speed = f
