from __future__ import absolute_import, unicode_literals, print_function

__author__ = 'nmearl'

import json
import os
import numpy as np
import pymultinest
import matplotlib.pyplot as plt
import utilfuncs
import photometry

x = None
y = None
yerr = None
rv_data = None
N = None
t0 = None
maxh = None
orbit_error = None
max_lnlike = -np.inf
fname = ""


def per_iteration(theta, lnl, model):
    global max_lnlike
    if lnl > max_lnlike:
        max_lnlike = lnl
        params = np.append(np.array([N, t0, maxh, orbit_error]), theta)
        params = utilfuncs.split_parameters(params)
        redchisqr = np.sum(((y - model) / yerr) ** 2) / (y.size - 1 - (N * 5 + (N - 1) * 6))
        utilfuncs.iterprint(params, lnl, redchisqr, 0.0, 0.0)
        utilfuncs.report_as_input(params, fname)


def lnprior(cube, ndim, nparams):
    theta = np.array([cube[i] for i in range(ndim)])
    sys = np.array([N, t0, maxh, orbit_error])
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = utilfuncs.split_parameters(np.append(sys, theta))[4:]

    masses = 10**(masses*8 - 9)
    radii = 10**(radii*4 - 4)
    fluxes = 10**(fluxes*4 - 4)
    a = 10**(a*2 - 2)
    e = 10**(e*3 - 3)
    inc *= 2.0 * np.pi
    om = 2.0 * np.pi * 10**(om*2 - 2)
    ln = 2.0 * np.pi * 10**(ln*8 - 8)
    ma = 2.0 * np.pi * 10**(ma*2 - 2)

    theta = np.concatenate([masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma])

    for i in range(ndim):
        cube[i] = theta[i]


def lnlike(cube, ndim, nparams):
    theta = np.array([cube[i] for i in range(ndim)])

    if len(theta[~np.isfinite(theta)]) > 0:
        return -np.inf

    sys = np.array([N, t0, maxh, orbit_error])
    params = utilfuncs.split_parameters(np.append(sys, theta))

    mod_flux, mod_rv = utilfuncs.model(params, x, rv_data[0])

    # lnf = np.log(1.0e-10)  # Natural log of the underestimation fraction
    # inv_sigma2 = 1.0 / (yerr ** 2 + model ** 2 * np.exp(2 * lnf))
    # lnl = -0.5 * (np.sum((y - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))

    mod_flux, mod_rv = utilfuncs.model(params, x, rv_data[0])
    flnl = np.sum((-0.5 * ((mod_flux - y) / yerr)**2))
    rvlnl = np.sum((-0.5 * ((mod_rv - rv_data[1]) / rv_data[2])**2))

    per_iteration(theta, flnl, mod_flux)

    return flnl + rvlnl


def generate(params, lx, ly, lyerr, lrv_data, lncores, lfname):
    lN, lt0, lmaxh, lorbit_error, masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = params

    global x, y, yerr, rv_data, N, t0, maxh, orbit_error, ncores, fname
    x, y, yerr, rv_data, N, t0, maxh, orbit_error, ncores, fname \
        = lx, ly, lyerr, lrv_data, lN, lt0, lmaxh, lorbit_error, lncores, lfname

    # number of dimensions our problem has
    parameters = ["{0}".format(i) for i in range(N*5 + (N-1)*6)]
    n_params = len(parameters)

    # make sure the output directories exist
    if not os.path.exists("./output"):
        os.mkdir("./output")

    if not os.path.exists("./output/{0}".format(fname)):
        os.mkdir("./output/{0}".format(fname))

    if not os.path.exists("./output/{0}/reports".format(fname)):
        os.mkdir("./output/{0}/reports".format(fname))

    if not os.path.exists("./output/{0}/plots".format(fname)):
        os.mkdir("./output/{0}/plots".format(fname))

    # we want to see some output while it is running
    # progress_plot = pymultinest.ProgressPlotter(n_params=n_params, outputfiles_basename='./output/{0}/reports/'.format(fname))
    # progress_plot.start()
    # progress_print = pymultinest.ProgressPrinter(n_params=n_params, outputfiles_basename='./output/{0}/reports/'.format(fname))
    # progress_print.start()

    # run MultiNest
    pymultinest.run(lnlike, lnprior, n_params, outputfiles_basename='./output/{0}/reports/'.format(fname),
                    resume=True, verbose=True)

    # run has completed
    # progress_plot.stop()
    # progress_print.stop()
    json.dump(parameters, open('./output/{0}/params.json'.format(fname), 'w'))  # save parameter names

    # plot the distribution of a posteriori possible models
    plt.figure()
    plt.plot(x, y, '+ ', color='red', label='data')
    a = pymultinest.Analyzer(outputfiles_basename="./output/{0}/reports/".format(fname), n_params=n_params)

    for theta in a.get_equal_weighted_posterior()[::100, :-1]:
        sys = np.array([N, t0, maxh, orbit_error])
        params = utilfuncs.split_parameters(np.append(sys, theta))

        mod_flux, mod_rv = utilfuncs.model(params, x)

        plt.plot(x, mod_flux, '-', color='blue', alpha=0.3, label='data')

    utilfuncs.report_as_input(params, fname)

    plt.savefig('./output/{0}/plots/posterior.pdf'.format(fname))
    plt.close()