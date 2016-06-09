import numpy as np

def general_residuals(xs, xbar, xbarold, rho):
    """ Compute the residuals for floats or numpy arrays.
    Suffers heavy overhead from np.linalg.norm in the case of all the
    data being floats.
    """
    npnorm = np.linalg.norm
    r = 0.0
    s = 0.0

    for x in xs:
        for k,v in x.items():
            xbark = xbar[k]
            r += npnorm(v - xbark)**2
            s += npnorm(xbark - xbarold[k])**2

    return np.sqrt(r), rho*np.sqrt(s)

def general_residuals2(xs, xbar, xbarold, rho):
    """ Reduces the overhead from `general_residuals()`,
    but not as much as `float_residuals()`.
    """

    npnorm = np.linalg.norm
    r = 0.0
    s = 0.0

    for x in xs:
        for k,v in x.items():
            xbark = xbar[k]

            rval = v - xbark
            sval = xbark - xbarold[k]

            if isinstance(rval, Number):
                r += (rval)**2
                s += (sval)**2
            else:
                r += npnorm(rval)**2
                s += npnorm(sval)**2

    return np.sqrt(r), rho*np.sqrt(s)

def float_residuals(xs, xbar, xbarold, rho):
    """ Compute the residuals when all the values in the dictionaries
    are floats (no numpy arrays allowed).

    Much faster than having to call np.linalg.norm, or check if the values
    are floats.

    XXX: have to update the algorithm to use this by default if eligible
    """
    r = 0.0
    s = 0.0

    for x in xs:
        for k,v in x.items():
            xbark = xbar[k]
            r += (v - xbark)**2
            s += (xbark - xbarold[k])**2

    return np.sqrt(r), rho*np.sqrt(s)