import time
from itertools import repeat

def map_apply(funcs, *iterables, rep_args=None, mapper=None):
    """ Apply each func to each iterable input argument.

    For coordinating proxes with their appropriate input.

    rep_args gives constant arguments to repeat across function calls (like rho).
    mapper gives the mapping function to use (concurrent.futures).

    returns the output results, along with the time for each function call
    """

    if mapper is None:
        mapper = map
    if rep_args is not None:
        rep_args = [repeat(i) for i in rep_args]
    else:
        rep_args = []

    out = mapper(do, funcs, *iterables, *rep_args)
    results, times = unzip(out)

    return results, times

def do(func, *iterables):
    # todo, put into timing module
    start = time.time()
    result =  func(*iterables)
    total_time = time.time() - start

    return result, total_time

def unzip(seq):
    """ Want unzip to return a tuple of lists
    """
    out = zip(*seq)
    out = tuple(list(i) for i in out)
    return out