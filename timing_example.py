from admm import ADMM, form_sharing_prox

from admm.tests.iter_times import agent_dict, make_dummy_prox

import numpy as np

p = 2
threads = None
filename = 'test_time.json'

n = 10**p # num goods
m = 10**p # num agents
k = 20 # goods per agent

B = {g: 0 for g in range(n)}

dicts = [agent_dict(n,k,i) for i in range(m)]
proxes = [make_dummy_prox(d) for d in dicts] + [form_sharing_prox(B)]



admm = ADMM(proxes, rho=1.0, threads=threads)
admm.step(10)

extra_info = dict(threads=threads,n=n,m=m,k=k)

admm.saveinfo(filename, extra_info)