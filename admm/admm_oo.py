from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt

from .admm import admm_step
from .timer import SimpleTimer
from .rho_adjust import make_resid_gap
from .report import report_solve, plot_iter_breakdown
from .resid import general_residuals, float_residuals

import json

import numpy as np


class ADMM:
    """ Maintains state of ADMM iteration.

    xbar, us maintain global and local ADMM state.
    proxes maintains the list of prox functions
    infos is the ADMM status info for each iteration
    timed_runs gives the runtime for each chunk of runs
    """
    def __init__(self, proxes, rho, rho_adj=None, hook=None, threads=None,
                 resid='general'):
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

        if resid == 'general':
            self._resid = general_residuals
        elif resid == 'float':
            self._resid = float_residuals
        else:
            raise ValueError('Unrecognized residual calculation method: {}'.format(resid))

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
                                rho_adj=self.rho_adj,
                                residuals=self._resid)

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

    def saveinfo(self, filename, extra=None):
        """ Save the descriptive stats of the ADMM iteration.

        Parameters
        ----------
        extra : dict
            A dictionary of extra info (possibly about the problem being solved)
            to save in the output json file.
        """
        data = dict(extra=extra)

        data['solve_time'] = sum(run[1] for run in self.timed_runs)
        data['infos'] = self.infos

        with open(filename, "w") as file:
            json.dump(data, file)

    def run_until_hook(self, max_iters=5000, tol=1e-3, substeps=100):
        """ run until hook <= tol
        """
        total_steps = 0
        header = '{:>6}  {:>8}  {:>8}'.format('Iter', 'Hook', 'Time (s)')
        print(header)

        while True:
            if total_steps >= max_iters:
                break

            msg = '{:6d}'.format(total_steps)
            print(msg, end='')

            with SimpleTimer() as t:
                self.step(substeps)

            hook_val = self.infos[-1]['hook']
            print('  {:8.2e}  {:8.2e}'.format(hook_val, t.time))

            total_steps += substeps

            if hook_val <= tol:
                break



def load(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    admm = ADMM([], 1.0)
    admm.infos = data['infos']

    return admm, data


        



