import cvxpy as cvx
import numpy as np

from . import coaverage


def form_agent_constr0(A, logB, x, phi):
    """ This formulation seems to reduce the number of
    variables, constraints, and NNZ in the A matrix.
    """
    m,n = A.shape
    constr = []
    
    for i in range(m):
        logcash = cvx.log_sum_exp(phi + logB[i,:])
        ag_exp = cvx.log(x[i,:]*A[i,:]) - logcash
        t = cvx.Variable()
        
        constr += [ag_exp >= t]
        for j in range(n):
            expr = t >= np.log(A[i,j]) - phi[j]
            constr += [expr]
    
    return constr

def solve_linear_eq(A,B):
    m,n = A.shape
    x = cvx.Variable(m,n)
    phi = cvx.Variable(n)
    
    logB = np.log(B)
    
    # todo: maybe mess with equality
    constr = [x >= 0, cvx.sum_entries(x,0).T == np.sum(B,0)]
    
    constr += form_agent_constr0(A, logB, x, phi)
            
    prob = cvx.Problem(cvx.Maximize(0), constr)
    prob.solve(verbose=True, solver='ECOS')
    
    
    return np.array(x.value), np.array(phi.value).flatten()

def alloc_violation(x,B):
    neg = sum(x[x<0])
    
    totalx = np.sum(x,0)
    totalB = np.sum(B,0)
    
    over = sum(np.maximum(totalx - totalB,0))

    return neg, over
    
def clean_x(x, B):
    x = x.copy()
    x[x<0] = 0
    
    total_x = np.sum(x,0)
    B = np.sum(B,0)
    
    over = np.maximum(total_x/B,1)
    
    x = x/over
    
    return x

def util(A,x):
    return np.sum(A*x, 1)

def max_util(a,b,phi):
    """Agent just goes all in on one item. Doesn't worry about global supply."""
    
    p = np.exp(phi)
    bangperbuck = a/p
    
    maxbpb = np.max(bangperbuck)
    cash = np.dot(b,p)
    
    return maxbpb*cash

def all_max_utils(A,B,phi):
    """ Get max utility at the current log-prices, phi,
    for each agent.
    """
    m,n = A.shape
    utils = []
    for i in range(m):
        utils += [max_util(A[i,:], B[i,:], phi)]
        
    return np.array(utils)

def displeasure(A,B,x,phi):
    m = all_max_utils(A,B,phi)
    a = util(A,x)
    
    return np.maximum((m-a)/m,0)



def foo_prox(a, b, x0, phi0, rho):
    n = len(a)
    x = cvx.Variable(n)
    phi = cvx.Variable(n)
    
    logb = np.log(b)
    
    logcash = cvx.log_sum_exp(phi + logb)
    ag_exp = cvx.log(x.T*a) - logcash
    t = cvx.Variable()

    constr = [x >= 0, ag_exp >= t]
    for j in range(n):
        expr = t >= np.log(a[j]) - phi[j]
        constr += [expr]
        
    obj = cvx.sum_squares(x-x0) + cvx.sum_squares(phi-phi0)
    obj = obj*rho/2.0
    prob = cvx.Problem(cvx.Minimize(obj), constr)
    prob.solve(verbose=False, solver='ECOS')
    
    return np.array(x.value).flatten(), np.array(phi.value).flatten()



def make_prox(a,b,agent):
    n = len(a)
    def prox(d, rho):
        
        if (agent,'x') not in d:
            x0 = np.zeros(n)
        else:
            x0 = d[(agent,'x')]
            
        if 'phi' not in d:
            phi0 = np.zeros(n)
        else:
            phi0 = d['phi']
            
        x, phi = foo_prox(a,b,x0,phi0,rho)
        
        d = {}
        d[(agent,'x')] = x
        d['phi'] = phi
        return d, None

    return prox
    
def second(item):
    return item[1]

    
def make_sharing_prox(total):
    """ total is a dict of the keys and total values interested
    
    Note: has to work with empty dict
    actually, i'm not sure if this can easily work with an empty dict
    maybe i can just skip the first iteration...
    """
    
    def foo(x0, rho=None):
        # can probably write a loop to go through dict instead of making new one
        x = {k:v for k,v in x0.items() if isinstance(k, tuple)}
        avg = coaverage.coaverage(total, key_transform=second)
        avg.send(x)
        
        xarrow = avg.send(None)
        
        # gotta adjust the input x0 values
        
        xout = {}
        for (agent, key), value in x.items():
            xout[(agent,key)] = x[(agent,key)] - xarrow[key]
        
        return xout, None
    
    return foo

def make_displeasure_hook(A,B):
    def displeasure_hook(xbar):
        dist_x = [(*k,v) for k,v in xbar.items() if isinstance(k,tuple)]
        dist_x = sorted(dist_x)
        dist_x = [x[2] for x in dist_x]
        x = np.array(dist_x)
        phi = xbar['phi']

        x = clean_x(x,B)

        return max(displeasure(A,B,x,phi))

    return displeasure_hook


        