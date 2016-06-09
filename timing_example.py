from admm import ADMM, form_sharing_prox

from admm.tests.iter_times import agent_dict, make_dummy_prox

import numpy as np

def run_experiment(n=0, m=0, k=0, seed=0, filename='test_time.json',
                   threads=None):

    np.random.seed(seed)
    B = {g: 0 for g in range(n)}

    dicts = [agent_dict(n,k,i) for i in range(m)]
    proxes = [make_dummy_prox(d) for d in dicts] + [form_sharing_prox(B)]

    admm = ADMM(proxes, rho=1.0, threads=threads, resid='float')
    admm.step(10)

    extra = dict(threads=threads,n=n,m=m,k=k, seed=seed, filename=filename)

    admm.saveinfo(filename, extra)

p = 3
threads = None
filename = 'test_time.json'

n = 10**p # num goods
m = 10**p # num agents
k = 20 # goods per agent

run_experiment(n, m, k, seed=0, filename=filename, threads=threads)