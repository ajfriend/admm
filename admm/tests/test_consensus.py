import cvxpy as cvx
import numpy as np
from scsprox import Prox

from admm import ADMM

def test_consensus():
    n = 100
    k = 4
    np.random.seed(0)

    proxes = []
    for _ in range(k):
        A = np.random.randn(n,n)
        b = np.random.randn(n)
        x = cvx.Variable(n)
        
        obj = cvx.Minimize(cvx.norm(A*x-b))
        prob = cvx.Problem(obj)
        
        prox = Prox(prob, {'x':x})
        proxes += [prox]

    admm = ADMM(proxes, rho=1)
    admm.step(100)

    # check that the residuals are relatively small
    assert 1e-7 <= admm.infos[-1]['r'] <= 1e-4
    assert 1e-7 <= admm.infos[-1]['s'] <= 1e-4

    # a few more admm steps should make the residuals very small
    admm.step(100)

    assert admm.infos[-1]['r'] <= 1e-8
    assert admm.infos[-1]['s'] <= 1e-8