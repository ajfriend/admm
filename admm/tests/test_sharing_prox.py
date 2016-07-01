from admm.sharing import sharing_prox
import numpy as np

def test_empty():
    B = {'apples': 2, 'oranges': 4}
    assert sharing_prox({},B) == {}


def test_simple_calc():
    B = {'apples': 2, 'oranges': 4}
    x = {('alice', 'apples'): 4, ('bob', 'apples'): 1, 'apples': 17, ('alice', 'oranges'): 3}
    assert sharing_prox(x,B) == {('alice', 'apples'): 2.5, ('alice', 'oranges'): 4.0, ('bob', 'apples'): -0.5}


def test_immutable_B():
    B = {'apples': np.zeros(2)}
    x = {(1, 'apples'): np.array([1,3]), (2, 'apples'): np.array([5,2])}

    sharing_prox(x,B)

    # make sure that B has not been modified
    assert np.all(B['apples'] == np.zeros(2))