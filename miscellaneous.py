import numpy as np


class Call:
    def __init__(self, s, e, sp, ep, start_time, wait_end_time, price, duration):
        """
        :param s: Start road index.
        :param e: End road index.
        :param sp: Start road position.
        :param ep: End road position.
        :param start_time: Call start time.
        :param wait_end_time: Wait end time.
        :param price: Price of the call.
        :param duration: Duration of the call.
        """
        self.s = s
        self.sp = sp
        self.e = e
        self.ep = ep
        self.start_time = start_time
        self.served_time = None
        self.wait_end_time = wait_end_time
        self.price = price
        self.duration = duration


class Road:
    def __init__(self, uuid=0, **kwargs):
        self.uuid = uuid
        self.length = kwargs['length'] # meter
        self.drivers = []
        self.calls = []
        # u, v is an id of start, end node.
        self.u = kwargs['u'].item()
        self.v = kwargs['v'].item()
        self.reachable_roads = []

        self.popularity = abs(np.random.normal(0, 1)) ** 3 / 20
        self.speed = 24 # km/h
        self.speed_coefficient = 1
        self.actionable_driver_number = 0

        temp = kwargs.get('speed_info_closest_road_index', None)
        if temp is None:
            temp_v = 0
        else:
            temp_v = temp.item()
        self.speed_info_closest_road_index = temp_v


class Driver:
    def __init__(self, uuid, road_index, road_position):
        self.uuid = uuid
        self.last_road_index = road_index
        self.road_index = road_index
        self.road_position = road_position
        self.current_serving_call = None
        self.movable_time = 0

    def is_online(self):
        return self.current_serving_call is not None

    def assign_call(self, call: Call):
        self.current_serving_call = call
