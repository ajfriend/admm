from itertools import repeat
from collections import defaultdict
from functools import wraps

from .timer import SimpleTimer, DictTimer

"""
What if this thing already has an info? do we clobber it? overwrite `time`?
"""
def time_info(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        with DictTimer('time',) as info:
            out = f(*args, **kwds)
        wrapper.info = info
        return out
    
    wrapper.info = {'time': None}

    return wrapper


def map_apply(funcs, *iterables, rep_args=None, mapper=None):
    """ Apply each func to each iterable input argument.

    For coordinating proxes with their appropriate input.

    rep_args gives constant arguments to repeat across function calls (like rho).
    mapper gives the mapping function to use (concurrent.futures).

    returns the output results, along with the time for each function call
    as a list of tuples: [(output1, time1), ...]
    """

    if mapper is None:
        mapper = map
    if rep_args is not None:
        rep_args = [repeat(i) for i in rep_args]
    else:
        rep_args = []

    out = mapper(do, funcs, *iterables, *rep_args)

    return out

def do(func, *iterables):
    with SimpleTimer() as t:
        result =  func(*iterables)

    return result, t.time

def fast_avg(xs):
    """ Compute the average by key over the list of dictionaries, xs.
    """
    total = defaultdict(float)
    count = defaultdict(int)
    
    for x in xs:
        for k,v in x.items():
            count[k] += 1
            total[k] += v
                
    out = {}
    for k,v in count.items():
        out[k] = total[k]/v
            
    return out