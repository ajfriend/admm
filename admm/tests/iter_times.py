import numpy as np
from ..timer import SimpleTimer
from .. import form_sharing_prox

def goodfloat(population=100, size=10, offset=0.0, mult=1.0):
    """ Return dict of `size` items, where the items
    are drawn with replacement from a pool of `population` integers.

    The values are given by `mult` a + `offset`, where a is a [0,1]
    uniform random variable.
    """
    z = zip(np.random.randint(population,size=size), mult*np.random.rand(size)+offset)
    return dict(z)

def agent_dict(n, k, agentid):
    """ Dummy output of an agent prox.
    
    Returns a dict of (agentid, good_key) keys for shared goods,
    and `good_key` keys for consensus prices.
    """

    util = goodfloat(population=n, size=k)
    endow = goodfloat(population=n, size=k)
    
    d = endow.copy()
    d.update(util)
    for k in util:
        d[(agentid, k)] = 0
    
    return d

def rand_replace_vals(d):
    """ Return a new `dict` with the values
    replaced by random numbers.
    """    
    d = {k: np.random.rand() for k in d}
    
    return d

def make_dummy_prox(d):
    
    def foo(x0, rho):
        with SimpleTimer() as t:
            x = rand_replace_vals(d)

        return x, dict(time=t.time)
    
    return foo

def dummy_market_proxes(n=10, m=10, k=5):
    """ Form a dummy market eq problem where proxes do nothing
    but return random keys. Use this function for ADMM timing tests.

    Warning: Does not check that each good is present in at least one utility
    function and at least one endowment.

    Returns the dummy agent proxes, along with the sharing prox.

    Parameters
    ----------
    n : int
        Number of goods
    m : int
        Number of agents
    k : int
        Number of goods per agent
        (randomly selected goods with replacement, so could be less than k)

    """
    B = {g: 0 for g in range(n)}

    dicts = [agent_dict(n,k,i) for i in range(m)]
    proxes = [make_dummy_prox(d) for d in dicts] + [form_sharing_prox(B)]

    return proxes