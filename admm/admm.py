from collections import defaultdict
from . import coaverage
from .rho_adjust import resid_gap, rescale_rho_duals

import numpy as np

import matplotlib.pyplot as plt

""" the admm algo won't know anything about shared keys.
That happens entirely in the fusion center prox function.
"""

def make_xin(xbar, u):
    """ Make the input to the prox function, xbar-u
    assume u has the proper keys
    assume xbar[key] is 0 if key is not in xbar
    assume xbar is a defaultdict that goes to zero
    
    Expect xbar=u={} on the first iteration
    """
    x_in = {}
    
    # todo: this is a weird hack for if u is not present, but xbar is
    if not u:
        for k in xbar:
            x_in[k] = xbar[k]
    else:
        for k in u:
            x_in[k] = xbar[k] - u[k]
        
    return x_in

def update_u(u, x, xbar):
    """update u based on keys in x.
    Modifies u. returns nothing."""
    for k in x:
        u[k] = u[k] + x[k] - xbar[k]
        
        
def admm_step(proxes, xbar, us, rho):
    # prep the input to the prox
    xins = [make_xin(xbar, u) for u in us]
    
    # then we prox
    xs = [prox(xin, rho) for xin, prox in zip(xins, proxes)]
    
    # then we compute xbar
    xbarold = xbar
    xbar = coaverage.average(xs)
    
    # then we update the us
    for u,x in zip(us,xs):
        update_u(u,x,xbar)
        
    # maybe de-mean the us

    # compute residuals, update iteration info
    r,s = residuals(xs, xbar, xbarold, rho)

    # adjust rho?

        
    return xbar, us, r, s

def admm(proxes, rho, steps=10):
    xbar = defaultdict(float)
    us = [defaultdict(float) for _ in proxes]

    rs = []
    ss = []
    
    for _ in range(steps):
        xbar, us, r, s = admm_step(proxes, xbar, us, rho)
        scale = constant(r,s)#resid_gap(r,s)
        rho, us = rescale_rho_duals(rho, us, scale)
        rs += [r]
        ss += [s]
    
    return xbar, rs, ss

def plot_resid(r,s):
    n = len(r)
    plt.semilogy(range(n), r, range(n), s)
    plt.legend(['r', 's'])

def residuals(xs, xbar, xbarold, rho):
    r = 0.0
    s = 0.0

    for x in xs:
        for k in x:
            r += np.linalg.norm(x[k] - xbar[k])**2
            s += np.linalg.norm(xbar[k] - xbarold[k])**2

    return np.sqrt(r), rho*np.sqrt(s)


