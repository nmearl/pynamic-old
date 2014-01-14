__author__ = 'nmearl'

import numpy as np


def split_parameters(theta, N):
    theta = map(float, theta)
    sys_params, ind_params = theta[:5 * N], theta[5 * N:]

    masses, radii, fluxes, u1, u2 = [np.array(sys_params[i:N + i]) for i in range(0, len(sys_params), N)]
    a, e, inc, om, ln, ma = [np.array(ind_params[i:N - 1 + i]) for i in range(0, len(ind_params), N - 1)]

    # print_out([masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma], N)
    # for par in [masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma]:
    #     print(par)

    return masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma