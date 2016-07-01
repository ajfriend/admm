from collections import defaultdict
from .timer import SimpleTimer
from .functional import time_info

def sharing_prox(x, B):
    """ Projects shared values in x so that they sum to value in B.
    
    Parameters
    ----------
    x: dict
        Interested in keys of form (agent, good). Ignores any keys that are not tuples.
    B: dict
        Has keys of form `good`.
        
    Returns
    -------
    xout: dict
        Keys (agent, good) are such that keys for the same good sum to the value B[good].
        
    >>> B = {'apples': 2, 'oranges': 4}
    >>> x = {('alice', 'apples'): 4, ('bob', 'apples'): 1, 'apples': 17, ('alice', 'oranges'): 3}
    >>> sharing_prox(x,B) == {('alice', 'apples'): 2.5, ('alice', 'oranges'): 4.0, ('bob', 'apples'): -0.5}
    True
    
    >>> sharing_prox({},B)
    {}
    
    """
    
    total = defaultdict(float)
    count = defaultdict(int)
    
    for k in x:
        if isinstance(k, tuple):
            a, g = k
            count[g] += 1
            total[g] += x[k]
            
    for g in count:
        total[g] = (B[g] - total[g])/count[g]
            
    xout = {}
    for k in x:
        if isinstance(k, tuple):
            a,g = k
            xout[k] = x[k] + total[g]
        
    return xout


def form_sharing_prox(B):
    
    @time_info
    def foo(x0=None, rho=1):
        if x0 is None:
            x0 = {}
        return sharing_prox(x0, B)

    return foo
