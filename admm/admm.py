from collections import defaultdict
from .rho_adjust import rescale_rho_duals
from .timer import Timer, PrintTimer

from .functional import map_apply, fast_avg
from .resid import general_residuals, float_residuals

import numpy as np
from toolz import keyfilter
import matplotlib.pyplot as plt

from numbers import Number

# there is some weird startup cost to this call.
# it was throwing off timing, making the first calculation of residuals
# seemingly take longer than needed.
# calling it once here removes the extra overhead. super weird...
np.linalg.norm([0,.1])

""" the admm algo won't know anything about shared keys.
That happens entirely in the fusion center prox function.
"""

"""
Proxes should be a function (or callable) and have the form:

x = prox(x0, rho, **kwargs)

where x, x0 are dictionaries with the keys and values
the prox cares about. Proxes should be able to handle
an empty input dict, which would correspond to the proper 0 element.
To do this, they need to know the keys and datashapes they expect to work on.

proxes may maintain state to exploit caching and warm-starting.

Optionally, the prox can provide an info attribute
`prox.info` which would provide a dictionary of information.
It is OK for the prox to just be a function with no such attribute.
If `hasattr(prox, 'info') == False`, the ADMM algorithm will
just record `None` as the corresponding info.

While `prox.info` can provide arbitrary information, the ADMM
algorithm will look for keys `time` and `iters`, corresponding
to the most-recent prox computation time, and (if iterative) the
number of iterations performed to compute the prox.
"""

# todo: add a selector for the rho adjustment

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

def get_prox_infos(proxes, keys=None):
    if keys is None:
        keys = ['time', 'iter']

    pred = lambda x: x in keys

    out = []
    for prox in proxes:
        if hasattr(prox, 'info'):
            out += [keyfilter(pred, prox.info)]
        else:
            out += [None]

    return out
        
def admm_step(proxes, xbar, us, rho, hook=None, mapper=None, rho_adj=None,
              residuals=general_residuals):
    """ Does one ADMM iteration
    - x_i = prox(xbar - u_i)
    - u_i = u_i + x_i _ xbar

    Returns:
    xbar
    us
    info: dict with info, including residuals and timing

    """
    step_info = {}
    step_info['rho'] = rho

    with Timer(step_info, 'total_step'):
        # prep the input to the prox
        with Timer(step_info, 'x_in'):
            xins = [make_xin(xbar, u) for u in us]

        
        # then we prox
        # total time
        # custom info from the proxes
        # built in timing info on each prox
        with Timer(step_info, 'total_proxes'):
            out = map_apply(proxes, xins, rep_args=[rho], mapper=mapper)
            xs, step_info['times']['proxes'] = zip(*out)

        with Timer(step_info, 'prox_infos'):
            step_info['prox_infos'] = get_prox_infos(proxes)
        
        with Timer(step_info, 'xbar'):
            xbarold = xbar
            xbar = fast_avg(xs)
        
        with Timer(step_info, 'us'):
            for u,x in zip(us,xs):
                update_u(u,x,xbar)
            
        # maybe de-mean the us

        with Timer(step_info, 'resid'):
            # compute residuals, update iteration info
            r,s = residuals(xs, xbar, xbarold, rho)
            step_info['r'] = r
            step_info['s'] = s

        # adjust rho?
        with Timer(step_info, 'rho_scaling'):
            rho, us, step_info = do_scaling(rho_adj, step_info, us)

        if hook:
            with Timer(step_info, 'hook'):
                step_info['hook'] = hook(xbar)
        
    return xbar, us, rho, step_info

def admm(proxes, rho, steps=10, hook=None, rho_adj=None):
    xbar = defaultdict(float)
    us = [defaultdict(float) for _ in proxes]

    infos = []

    for _ in range(steps):
        xbar, us, rho, step_info = admm_step(proxes, xbar, us, rho, hook=hook, rho_adj=rho_adj)
    
        infos += [step_info]
    
    return xbar, infos


def do_scaling(scale_func, step_info, us):
    """ Rescale rho (and the us)
    as a result of the residual information (r,s)
    in `step_info`, and the stored rho value in
    `step_info`.

    Modify in place the step_info dict,
    but return it just to make explicit that
    it may be modified
    """
    r,s = step_info['r'], step_info['s']
    rho = step_info['rho']

    if scale_func:
        scale = scale_func(r,s)

    if scale != 1.0:
        rho, us = rescale_rho_duals(rho, us, scale)

    return rho, us, step_info
