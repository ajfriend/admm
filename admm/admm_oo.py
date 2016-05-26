from collections import defaultdict

import matplotlib.pyplot as plt

from .admm import admm_step, get_info
from .timer import SimpleTimer
from .rho_adjust import make_resid_gap


class ADMM:
    def __init__(self, proxes, rho, mapper=None, hook=None, rho_adj=None):
        self.mapper = mapper
        self.hook = hook

        if rho_adj is None:
            rho_adj = make_resid_gap()

        self.rho_adj = rho_adj

        self.rho = rho
        self.proxes = list(proxes)
        self.infos = []

        self.xbar = defaultdict(float)
        self.us = [defaultdict(float) for _ in proxes]

        self.timed_runs = []

    def step(self, num_steps=1):
        with SimpleTimer() as elapsed:
            for _ in range(num_steps):
                out = admm_step(self.proxes,
                                self.xbar,
                                self.us,
                                self.rho,
                                hook=self.hook,
                                mapper=self.mapper,
                                rho_adj=self.rho_adj)   

                self.xbar, self.us, self.rho, step_info = out     

                self.infos += [step_info]

        runtime = elapsed.time
        self.timed_runs += [ (num_steps, runtime) ]

    @property
    def total_time(self):
        time = 0.0
        for run in self.timed_runs:
            time += run[1]

        return time



    def plot_resid(self):
        r,s = get_info(self.infos, 'r', 's')
        n = len(r)
        plt.semilogy(range(n), r, range(n), s)
        plt.legend(['r', 's'])

    def plot_resid_info(self):
        r,s, disp = get_info(self.infos, 'r', 's', 'hook')
        n = len(r)
        plt.semilogy(range(n), r, range(n), s, range(n), disp)
        plt.legend(['r', 's', 'hook'])


