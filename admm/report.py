from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from toolz import get, get_in


def get_key(infos, *keys, reduce=None):
    """ Build array of ADMM logging output based on input key(s).
    These are info and timings done by the ADMM object (and not the prox objects).
    
    `keys` can be a single key, or several, nested keys
    
    - r, s, rho, hook: float
    - times: dict
        - proxes: list[float]
        - hook, resid, rho_scaling, total_proxes, total_step, us, x_in, xbar: float
    """
    # maybe just make a generator?
    g = (get_in(keys, info) for info in infos)
    
    if reduce:
        g = (reduce(row) for row in g)
    
    return np.array(list(g))

# todo: a minmax function reduces to two columns
def get_prox_key(infos, key, default=None, reduce=None, array=True):
    """ Build array of prox output for each operator and iteration.
    
    Return an {#iterations} by {#prox operators} array (unless a reduction is performed).
    
    reduce is usually, np.mean, np.min, or np.max
    
    Some prox outputs may be `None` or `{}`. In that case,
    return the `default` value.
    """
    g = ([get(key, p, default=default) for p in info['prox_infos']] for info in infos )
    
    if reduce:
        g = (reduce(row) for row in g)
        
    g = list(g)
    if array:
        g = np.array(g)

    return g


def get_prox_status(infos):
    # todo: if all one thing, can just return that key
    
    lam = lambda x: dict(Counter(x))
    out = get_prox_key(infos, 'status', reduce=lam, array=False)
    
    return out


def report_iters(infos, plot=True, ax=None):
    # todo: maybe we should put in zero for things without iters, so proxers line up
    # what about a reporting mode, to summarize?
    pred = lambda x: x >= 0
    f = lambda x: list(filter(pred, x))
    
    a = get_prox_key(infos, 'iter', default=-1, reduce=f)
    
    #a = [iters.max(axis=1), iters.mean(axis=1), iters.min(axis=1)]
    #a = np.stack(a, axis=1)

    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel('iteration')
    else:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    if plot:
        ax.plot(a, c='b', alpha=.3, linewidth=2.0)
        ax.set_ylabel('# prox iters')
        ax.set_title('Prox Iterations')
    
    return a
    

def report_rhos(infos, plot=True, ax=None):
    rhos = get_key(infos, 'rho')
    
    a = np.array(rhos)

    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel('iteration') 

    if plot:
        ax.semilogy(a, '-',basey=2, linewidth=2)
        
        ax.set_ylabel(r'$\rho$')
        ax.set_title(r'Varying $\rho$')
        ax.set_ylim([a.min()/2.0, a.max()*2.0])
    
    return a

def report_convergence(infos, hook=True, plot=True, ax=None):
    r = get_key(infos, 'r')
    s = get_key(infos, 's')
    
    vals = [r,s]
    labels = ['r', 's']
    if hook:
        h = get_key(infos, 'hook')
        vals += [h]
        labels += ['hook']
    
    a = np.stack(vals, axis=1)

    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel('iteration')

    if plot:
        ax.semilogy(a, linewidth=2)
        ax.legend(labels)
        ax.set_title('Residuals')
    
    return a

def report_time(infos, plot=True, inner=False, ax=None):
    """ Returns timing data (and optionally plots) for each iteration
    - total ADMM step time
    - total time for all prox computations (as timed/seen by ADMM alg)
    - sum of the individual prox computation times (as timed by ADMM)
    - sum of individual prox times (as reported by proxers)
    
    """
    total_step = get_key(infos, 'times', 'total_step')
    total_proxes = get_key(infos, 'times', 'total_proxes')
    prox_outers = get_key(infos, 'times', 'proxes', reduce=np.sum)
    prox_inners = get_prox_key(infos, 'time', default=0, reduce=np.sum)

    if inner:
        a = np.stack([total_step, total_proxes, prox_outers, prox_inners], axis=1)
        legend = ['full ADMM step', 'total_proxes', 'sum of prox times', 'prox_inners']
    else:
        a = np.stack([total_step, prox_outers], axis=1)
        legend = ['full ADMM step', 'sum of prox times']
    
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel('iteration')
    else:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    if plot:
        ax.plot(a, '-', linewidth=2)
        ax.legend(legend)
        ax.set_ylabel('time (s)')
        ax.set_title('ADMM Step Times')
    
    return a

def report_prox_time(infos, outer=True, plot=True, ax=None):
    """ Gets the prox times as measured by ADMM (outer), or as measured
    by the prox operator itsemf (inner).
    """
    # always defaulting to zero for time makes sense here (in a way that iters might not)
    if outer:
        title = 'Outer ProxOp Times'
        a = get_key(infos, 'times', 'proxes')
    else:
        title = 'Inner ProxOp Times'
        a = get_prox_key(infos, 'time', default=0)
        
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel('iteration')
    else:
        if outer == False:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

    if plot:
        ax.plot(a, c='b', alpha=.3, linewidth=2.0)
        ax.set_title(title)
        ax.set_ylabel('time (s)')

def report_solve(infos, figsize=(12,6), verbose=False):
    with plt.style.context(['seaborn-darkgrid', {'font.size': 12}]):
        if verbose is False:
            fig, ax = plt.subplots(2, 2, figsize=figsize, sharex=True)
            inner = False
        else:
            fig, ax = plt.subplots(3, 2, figsize=(12,9), sharex=True)
            inner = True

        _ = report_convergence(infos, hook=True, plot=True, ax=ax[0][0])
        _ = report_rhos(infos, plot=True, ax=ax[1][0])
        time = report_time(infos, inner=inner, ax=ax[0][1])
        _ = report_iters(infos, ax=ax[1][1])

        if verbose:
            _ = report_prox_time(infos, outer=True, plot=True, ax=ax[2][0])
            _ = report_prox_time(infos, outer=False, plot=True, ax=ax[2][1])

        for a in ax[-1]:
            a.set_xlabel('iteration')

        fig.tight_layout()

    step_time = time[:,0].sum()
    print('Total ADMM solve time: {:.2f} seconds'.format(step_time))

