
def make_resid_gap(gap=10.0, rho_adj=2.0, tol=1e-3):
    """ scale rho by factor `rho_adj` according to residual gap size `gap`.

    Don't do any scaling if both residuals are below `tol`.
    Changing rho at very low residuals changes the problems enough
    that we can see large jumps in the residuals. This is especially
    true when only doing approximate solves. When residuals
    are in range [0,tol], rho adjustment gives ADMM enough time to
    find approximately the right rho scaling.
    """
    mu = gap
    tau = rho_adj

    if tol is None:
        tol = 0

    def resid_gap(r, s):
        scale = 1.0

        if max(r,s) <= tol:
            return 1.0

        if r > mu*s:
            scale = tau
        elif s > mu*r:
            scale = 1.0/tau

        return scale

    return resid_gap

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