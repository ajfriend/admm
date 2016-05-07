
def make_resid_gap(gap=5.0, rho_adj=2.0):
    mu = gap
    tau = rho_adj

    def resid_gap(r, s):
        scale = 1.0
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