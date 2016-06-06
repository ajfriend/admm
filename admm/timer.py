import time
from contextlib import contextmanager

class Elapsed:
    def __init__(self, time):
        self.time = time


@contextmanager
def SimpleTimer():
    elapsed = Elapsed(None)
    start = time.time()
    try:
        yield elapsed
    finally:
        end = time.time()
        elapsed.time = end-start

@contextmanager
def DictTimer(label='time', d=None):
    if d is None:
        d = {}
    start = time.time()
    try:
        yield d
    finally:
        end = time.time()
        d[label] = end-start


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

@contextmanager
def PrintTimer(label):
    start = time.time()
    try:
        yield 
    finally:
        end = time.time()
        print('{}: {}'.format(label, end-start))