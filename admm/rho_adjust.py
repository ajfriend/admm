#def resid_gap(state, iter_info):
def resid_gap(r, s):
    mu = 5.0
    tau = 2.0

    scale = 1.0
    if r > mu*s:
        scale = tau
    elif s > mu*r:
        scale = 1.0/tau

    return scale

def constant(r,s):
    return 1.0


def rescale_rho_duals(rho, us, scale):
    """ Rescale rho and the duals by factor `scale`.
    """

    if scale != 1.0:
        rho *= scale

        for u in us:
            for k in u:
                u[k] /= scale

    return rho, us