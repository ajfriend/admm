from admm import ADMM, form_sharing_prox, load

from .iter_times import agent_dict, make_dummy_prox

import numpy as np

import warnings
import os


def test1():
    np.random.seed(0)

    n = 10 # num goods
    m = 10 # num agents
    k = 5 # goods per agent

    B = {g: 0 for g in range(n)}

    dicts = [agent_dict(n,k,i) for i in range(m)]
    proxes = [make_dummy_prox(d) for d in dicts] + [form_sharing_prox(B)]


    threads = None
    admm = ADMM(proxes, rho=1.0, threads=threads)

    admm.step(10)

    admm.iter_breakdown()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        admm.report(verbose=True)

    filename = '___test1.json'
    admm.saveinfo(filename, {'threads': threads})


    admm2, data = load(filename)

    assert data['threads'] == threads

    admm2.iter_breakdown()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        admm2.report(verbose=True)

    os.remove(filename)
