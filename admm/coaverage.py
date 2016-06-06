from collections import defaultdict
from functools import wraps

def coroutine(func):
    @wraps(func)
    def primer(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)
        return gen
    
    return primer

def identity(key):
    return key

@coroutine
def coaverage(offset=None, key_transform=None):
    """ Coroutine which computes running average of keys.

    Send in `dict`s to add to the value and count of each key.
    Send in `None` to compute and return the current average values.

    `offset` is the value to *subtract* from each key.

    >>> d1 = dict(a=1,b=1,c=1)
    >>> d2 = dict(b=2)
    >>> avg = coaverage()
    >>> avg.send(d1)
    >>> avg.send(d2)
    >>> result = avg.send(None)
    >>> result['a']
    1.0
    >>> result['b']
    1.5

    >>> d1 = dict(a=1,b=1)
    >>> d2 = dict(b=2)
    >>> avg = coaverage(offset=d1)
    >>> avg.send(d2)
    >>> avg.send(None)
    {'b': 1.0}
    >>> avg.send(d2)
    >>> avg.send(None)
    {'b': 1.5}
    """
    if key_transform is None:
        key_transform = identity

    total = defaultdict(float)
    count = defaultdict(int)
    
    if offset is not None:
        for k,v in offset.items():
            total[k] -= v
    
    avg = None
    
    while True:
        d = yield avg
        
        # if incoming value is None, compute and return average dict
        if d is None:
            avg = {}
            for key in count:
                avg[key] = total[key]/count[key]
        # if not None, expect a dictionary, and add to the running total and count
        # the corresponding `.send()` returns None
        else:
            for k, v in d.items():
                k = key_transform(k)
                total[k] += v
                count[k] += 1
                avg = None

def average(iterable, offset=None):
    """ take in an iterable of dictionaries. average along keys.
    """
    avg = coaverage(offset=offset)
    for item in iterable:
        avg.send(item)

    return avg.send(None)
