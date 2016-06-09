from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt

from .admm import admm_step, get_info
from .timer import SimpleTimer
from .rho_adjust import make_resid_gap
from .report import report_solve, plot_iter_breakdown

import json


class ADMM:
    """ Maintains state of ADMM iteration.

    xbar, us maintain global and local ADMM state.
    proxes maintains the list of prox functions
    infos is the ADMM status info for each iteration
    timed_runs gives the runtime for each chunk of runs
    """
    def __init__(self, proxes, rho, rho_adj=None, hook=None, threads=None):
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

        self.set_threads(threads)

    def step(self, num_steps=1):
        """ Perform `num_steps` ADMM steps and log results.
        """
        with SimpleTimer() as elapsed:
            for _ in range(num_steps):
                out = admm_step(self.proxes,
                                self.xbar,
                                self.us,
                                self.rho,
                                hook=self.hook,
                                mapper=self._mapper,
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

    def report(self, figsize=(12,6), hook=False, verbose=False):
        report_solve(self.infos, figsize=figsize, verbose=verbose, hook=hook)

    def iter_breakdown(self, iter_nums=None):
        plot_iter_breakdown(self.infos, iter_nums=iter_nums)

    def set_threads(self, threads):
        if threads is None or threads == 0:
            self._mapper = map
        else:
            ex = ThreadPoolExecutor(threads)
            self._mapper = ex.map

    def saveinfo(self, filename, extra_data=None):
        # num agents: nope
        data = {}

        data.update(extra_data)
        data['solve_time'] = sum(run[1] for run in self.timed_runs)
        data['infos'] = self.infos

        with open(filename, "w") as file:
            json.dump(data, file)  

def load(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    admm = ADMM([], 1.0)
    admm.infos = data['infos']

    return admm, data


        



