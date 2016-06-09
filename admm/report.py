from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from toolz import get, get_in

import pandas as pd

def plot_iter_breakdown(infos, iter_nums=None):
    keys = ['resid', 'rho_scaling', 'total_proxes', 'us', 'x_in', 'xbar']
    if iter_nums is None:
        # look at three most-recent iterations
        iters = [-3, -2,-1]

    rows = []

    for i in iters:
        out = {key: infos[i]['times'][key] for key in keys}
        if i >= 0:
            out['iter'] = i
        else:
            out['iter'] = len(infos)+i
    
        rows += [out]
    

    df = pd.DataFrame(rows)
    df = df.set_index('iter')
    
    fig, ax = plt.subplots()
    ax.set_title('ADMM iteration time breakdown')
    ax.set_ylabel('seconds')
    

    df.plot.bar(stacked=True, legend=True, ax=ax)
    ax.set_xlabel('iteration number')
    
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))


def get_key(infos, *keys, reduce=None, default=None):
    """ Build array of ADMM logging output based on input key(s).
    These are info and timings done by the ADMM object (and not the prox objects).
    
    `keys` can be a single key, or several, nested keys
    
    - r, s, rho, hook: float
    - times: dict
        - proxes: list[float]
        - hook, resid, rho_scaling, total_proxes, total_step, us, x_in, xbar: float
    """
    # maybe just make a generator?
    g = (get_in(keys, info, default=default) for info in infos)
    
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


def report_iters(infos, ax=None):
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

    if a: # a might be empty if no prox reports iterations
        ax.plot(a, c='b', alpha=.3, linewidth=2.0)
    ax.set_ylabel('# prox iters')
    ax.set_title('Prox Iterations')
    
    return a
    

def report_rhos(infos, ax=None):
    rhos = get_key(infos, 'rho')
    
    a = np.array(rhos)

    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel('iteration') 

    ax.semilogy(a, '-',basey=2, linewidth=2)
    
    ax.set_ylabel(r'$\rho$')
    ax.set_title(r'Varying $\rho$')
    ax.set_ylim([a.min()/2.0, a.max()*2.0])
    
    return a

def report_convergence(infos, hook=True, ax=None):
    # todo: interpolate for hook if only computed periodically
    r = get_key(infos, 'r')
    s = get_key(infos, 's')
    
    vals = [r,s]
    labels = ['r', 's']
    if hook:
        h = get_key(infos, 'hook', default=0)
        vals += [h]
        labels += ['hook']
    
    a = np.stack(vals, axis=1)

    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel('iteration')

    ax.semilogy(a, linewidth=2)
    ax.legend(labels)
    ax.set_title('Residuals')
    
    return a

def report_time(infos, inner=False, ax=None):
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
        legend = ['full ADMM step', 'prox "map" time', 'sum of prox times', 'self-reported prox times']
    else:
        a = np.stack([total_step, prox_outers], axis=1)
        legend = ['full ADMM step', 'sum of prox times']
    
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel('iteration')
    else:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    ax.plot(a, '-', linewidth=2)
    ax.legend(legend)
    ax.set_ylabel('time (s)')
    ax.set_title('ADMM Step Times')
    
    return a

def report_prox_time(infos, outer=True, ax=None):
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

    ax.plot(a, c='b', alpha=.3, linewidth=2.0)
    ax.set_title(title)
    ax.set_ylabel('time (s)')

def report_solve(infos, figsize=(12,6), hook=False, verbose=False):
    # todo: share the y axis on the bottom row if verbose==True
    with plt.style.context(['seaborn-darkgrid', {'font.size': 12}]):
        if verbose is False:
            fig = plt.figure(figsize=figsize)
            row1 = [plt.subplot(221), plt.subplot(222)]
            row2 = [plt.subplot(223), plt.subplot(224)]
            ax = [row1, row2]

            #fig, ax = plt.subplots(2, 2, figsize=figsize, sharex=True)
            inner = False
        else:
            fig = plt.figure(figsize=(12,9))
            row1 = [plt.subplot(321), plt.subplot(322)]
            row2 = [plt.subplot(323), plt.subplot(324)]

            ax1 = plt.subplot(325)
            row3 = [ax1, plt.subplot(326,sharey=ax1)]

            ax = [row1, row2, row3]
            #fig, ax = plt.subplots(3, 2, figsize=(12,9), sharex=True)
            inner = True

        _ = report_convergence(infos, hook=hook, ax=ax[0][0])
        _ = report_rhos(infos, ax=ax[1][0])
        time = report_time(infos, inner=inner, ax=ax[0][1])
        _ = report_iters(infos, ax=ax[1][1])

        if verbose:
            _ = report_prox_time(infos, outer=True, ax=ax[2][0])
            _ = report_prox_time(infos, outer=False, ax=ax[2][1])

        for a in ax[-1]:
            a.set_xlabel('iteration')

        fig.tight_layout()

    step_time = time[:,0].sum()
    print('Total ADMM solve time: {:.2f} seconds'.format(step_time))

