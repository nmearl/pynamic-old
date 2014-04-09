__author__ = 'nmearl'

import json
import sys
import numpy as np
import scipy.stats, scipy
import pymultinest
import matplotlib.pyplot as plt
import utilfuncs
import photometry

x = None
y = None
yerr = None
N = None
t0 = None
maxh = None
orbit_error = None
ncores = 1


def per_iteration(params, lnl, model):
    redchisqr = np.sum(((y - model) / yerr) ** 2) / (y.size - 1 - (N * 5 + (N - 1) * 6))
    utilfuncs.iterprint(N, params, lnl, redchisqr, 0.0, 0.0)
    utilfuncs.report_as_input(N, t0, maxh, orbit_error,  utilfuncs.split_parameters(params, N), 'multinest')


def lnprior(cube, ndim, nparams):
    theta = np.array([cube[i] for i in range(ndim)])
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = utilfuncs.split_parameters(theta, N)

    masses[(masses <= 0.0) | (masses > 0.1)] = -np.inf
    radii[(radii <= 0.0) | (radii > 1.0)] = -np.inf
    fluxes[(fluxes > 1.0) | (fluxes < 0.0)] = -np.inf
    u1[(u1 > 1.0) | (u1 < 0.0)] = -np.inf
    u2[(u2 > 1.0) | (u2 < 0.0)] = -np.inf
    a[(a < 0.0) | (a > 100.0)] = -np.inf
    e[(e > 1.0) | (e < 0.0)] = -np.inf
    inc[(inc > 2.0 * np.pi) | (inc < 0.0)] = -np.inf
    om[(om > (2.0 * np.pi)) | (om < -(2.0 * np.pi))] = -np.inf
    ln[(ln > (2.0 * np.pi)) | (ln < -(2.0 * np.pi))] = -np.inf
    ma[(ma > (2.0 * np.pi)) | (ma < 0.0)] = -np.inf

    theta = np.concatenate([masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma])

    for i in range(ndim):
        cube[i] = theta[i]


def lnlike(cube, ndim, nparams):
    theta = np.array([cube[i] for i in range(ndim)])

    if len(theta[~np.isfinite(theta)]) > 0:
        return -np.inf

    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = utilfuncs.split_parameters(theta, N)

    model = photometry.multigenerate(ncores,
        N, t0, maxh, orbit_error,
        x,
        masses, radii, fluxes, u1, u2,
        a, e, inc, om, ln, ma
    )

    # lnf = np.log(1.0e-10)  # Natural log of the underestimation fraction
    # inv_sigma2 = 1.0 / (yerr ** 2 + model ** 2 * np.exp(2 * lnf))
    # lnl = -0.5 * (np.sum((y - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))

    lnl = (-0.5 * ((model - y) / yerr)**2).sum()

    per_iteration(theta, lnl, model)

    return lnl


def generate(params, lx, ly, lyerr, lncores=1):
    lN, lt0, lmaxh, lorbit_error, masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = params

    global x, y, yerr, N, t0, maxh, orbit_error, ncores
    x, y, yerr, N, t0, maxh, orbit_error, ncores = lx, ly, lyerr, lN, lt0, lmaxh, lorbit_error, lncores

    # analyse the file given as first argument
    datafile = "test"

    # analyse with 1 gaussian

    # number of dimensions our problem has
    parameters = ["{0}".format(i) for i in range(N*5 + (N-1)*6)]
    n_params = len(parameters)

    # run MultiNest
    pymultinest.run(lnlike, lnprior, n_params, outputfiles_basename=datafile + '_1_', resume=False, verbose=True)
    json.dump(parameters, open(datafile + '_1_params.json', 'w'))  # save parameter names

    # plot the distribution of a posteriori possible models
    plt.figure()
    plt.plot(x, y, '+ ', color='red', label='data')
    a = pymultinest.Analyzer(outputfiles_basename=datafile + '_1_', n_params=n_params)
    for theta in a.get_equal_weighted_posterior()[::100, :-1]:
        masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = utilfuncs.split_parameters(theta, N)
        plt.plot(
            x,
            photometry.generate(N, t0, maxh, orbit_error, x, masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma),
            '-', color='blue', alpha=0.3, label='data'
        )

    plt.savefig(datafile + '_1_posterior.pdf')
    plt.close()

    a_lnZ = a.get_stats()['global evidence']