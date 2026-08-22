import numpy as np


class Meter(object):
    def reset(self):
        pass

    def add(self, value, n=1):
        pass

    def value(self):
        pass


class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.n = 0
        self.mean = 0
        self.val = 0
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0
        self.val = 0

    def add(self, value, n=1):
        if n <= 0:
            raise ValueError('n must be positive')
        self.val = value
        total = self.n + n
        self.mean = (self.mean * self.n + value * n) / total
        self.n = total

