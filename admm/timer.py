import time
from contextlib import contextmanager


@contextmanager
def Timer(d, label):
    start = time.time()
    try:
        yield 
    finally:
        end = time.time()
        if 'times' not in d:
            d['times'] = {}
        d['times'][label] = end-start
