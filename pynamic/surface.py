__author__ = 'nmearl'

import numpy as np
import modeler
from itertools import product
from multiprocessing import Process, Queue, Pool
import os
import pylab


def generate(params, n, x, y, yerr, N):
    # par_ranges = [
    #     # np.random.normal(par.value, 1.0, n)
    #     np.linspace(par.value - 0.1*par.value, par.value + 0.1*par.value, n)
    #     if par.vary else np.array([par.value, ])
    #     for name, par in params.items()
    # ]

    # vary_par = param_item[1][0]

    par_ranges = [
        # np.random.normal(par.value, 1.0, n)
        np.array([params[i], ])
        if not i == 4 + N else np.linspace(params[i] * 0.8, params[i] * 1.2, n)
        for i in range(len(params))
    ]

    par_grid = product(*par_ranges)

    print "Allocating process pool..."
    process(par_grid, x, y, yerr, N)


def process(par_grid, x, y, yerr, N):
    pool = Pool(processes=1)

    for par_set in par_grid:
        #if not os.path.exists('/home/nearl/kepler/cached/{0}.npz'.format(kep_id)):
        # pool.apply_async(chi_surface, args=(par_set, x, y, yerr), callback=write_out)
        results = chi_surface(par_set, x, y, yerr)
        write_out(results)
        # p = Process(target=get_data, args=(kep_dir, kep_id, queue))
        # jobs.append(p)
        # p.start()

    pool.close()
    pool.join()


def write_out(results):
    if not any(results):
        return

    cache_path = './output/chi_surface.dat'

    # np.savez(cache_path, masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma, chi)

    status = 'a' if os.path.exists(cache_path) else 'w'

    with open(cache_path, status) as f:
        for val in results:
            f.write('{0} '.format(val))

        f.write('\n')


def _split_parameters(theta, N):
    sys_params, ind_params = theta[:5 * N], theta[5 * N:]

    masses, radii, fluxes, u1, u2 = [np.array(sys_params[i:N + i]) for i in range(0, len(sys_params), N)]
    a, e, inc, om, ln, ma = [np.array(ind_params[i:N - 1 + i]) for i in range(0, len(ind_params), N - 1)]

    return masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma


def chi_surface(par_set, x, y, yerr):
    N, t0, maxh, orbit_error = par_set[:4]

    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = _split_parameters(par_set[4:], int(N))

    model = modeler.generate(
        N, t0, maxh, orbit_error,
        x,
        masses, radii, fluxes, u1, u2,
        a, e, inc, om, ln, ma
    )

    # pylab.plot(x, y, 'k')
    # pylab.plot(x, model, 'r+')
    # pylab.show()

    chi2 = np.sum(((y - model) / yerr) ** 2)
    print chi2
    results = np.append(par_set, chi2)

    return results