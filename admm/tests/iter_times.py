import numpy as np

def goodfloat(population=100, size=10, offset=0.0, mult=1.0):
    z = zip(np.random.randint(population,size=size), mult*np.random.rand(size)+offset)
    return dict(z)

def agent_dict(n, k, agentid):
    util = goodfloat(population=n, size=k)
    endow = goodfloat(population=n, size=k)
    
    d = endow.copy()
    d.update(util)
    for k in util:
        d[(agentid, k)] = 0
    
    return d

def rand_replace_vals(d):        
    d = {k: np.random.rand() for k in d}
    
    return d

def make_dummy_prox(d):
    
    def foo(x0, rho):
        x = rand_replace_vals(d)
        
        return x, None
    
    return foo