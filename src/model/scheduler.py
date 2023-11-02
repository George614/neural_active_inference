import numpy as np
import math

"""
Various learning rate schedule modules used in reinforcement learning.
Inspired by https://github.com/higgsfield/RL-Adventure
@author: Zhizhuo (George) Yang
"""


class Schedule(object):
    def __init__(self, start=0.0, stop=1.0):
        self.start = start
        self.stop = stop


class Linear_schedule(Schedule):
    def __init__(self, start=0.0, stop=1.0, duration=300):
        super().__init__(start, stop)
        self.duration = duration

    def __call__(self, idx):
        val = self.start + (self.stop - self.start) * \
            np.minimum(idx / self.duration, 1.0)
        return val


class Exponential_schedule(Schedule):
    def __init__(self, start=0.0, stop=1.0, decay=300):
        super().__init__(start, stop)
        self.decay = decay

    def __call__(self, idx):
        val = self.stop + (self.start - self.stop) * \
            math.exp(-1.0 * idx / self.decay)
        return val


class Frange_cycle_linear(Schedule):
    ''' Cyclical Annealing Schedule: A Simple Approach to Mitigating {KL}
    Vanishing. Fu etal NAACL 2019 '''

    def __init__(self, n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
        super().__init__(start, stop)
        self.n_cycle = n_cycle
        self.ratio = ratio
        self.n_iter = n_iter

    def __call__(self):
        L = np.ones(self.n_iter) * self.stop
        period = self.n_iter / self.n_cycle
        step = (self.stop - self.start) / \
            (period * self.ratio)  # linear schedule

        for c in range(self.n_cycle):
            v, i = self.start, 0
            while v <= self.stop and (int(i + c * period) < self.n_iter):
                L[int(i + c * period)] = v
                v += step
                i += 1
        return L
