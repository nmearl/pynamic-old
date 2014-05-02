from __future__ import absolute_import, unicode_literals, print_function

__author__ = 'nmearl'

import json
import os
import numpy as np
import pymultinest
import matplotlib.pyplot as plt
import utilfuncs

max_lnlike = -np.inf
mod_pars = None
photo_data = None
rv_data = None
ncores = 1
fname = ""


def per_iteration(mod_pars, theta, lnl, model):
    global max_lnlike
    if lnl > max_lnlike:
        max_lnlike = lnl
        params = utilfuncs.split_parameters(theta, mod_pars[0])
        redchisqr = np.sum(((photo_data[1] - model) / photo_data[2]) ** 2) / \
                    (photo_data[1].size - 1 - (mod_pars[0] * 5 + (mod_pars[0] - 1) * 6))

        utilfuncs.iterprint(mod_pars, params, max_lnlike, redchisqr, 0.0, 0.0)
        utilfuncs.report_as_input(mod_pars, params, fname)


def lnprior(cube, ndim, nparams):
    theta = np.array([cube[i] for i in range(ndim)])
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = utilfuncs.split_parameters(theta, mod_pars[0])

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

    params = utilfuncs.split_parameters(theta, mod_pars[0])

    mod_flux, mod_rv = utilfuncs.model(mod_pars, params, photo_data[0], rv_data[0], ncores)

    flnl = np.sum((-0.5 * ((mod_flux - photo_data[1]) / photo_data[2]) ** 2))
    rvlnl = np.sum((-0.5 * ((mod_rv - rv_data[1]) / rv_data[2])**2))

    per_iteration(mod_pars, theta, flnl, mod_flux)

    return flnl + rvlnl


def generate(lmod_pars, lparams, lphoto_data, lrv_data, lncores, lfname):
    global mod_pars, params, photo_data, rv_data, ncores, fname
    mod_pars, params, photo_data, rv_data, ncores, fname = \
        lmod_pars, lparams, lphoto_data, lrv_data, lncores, lfname

    # number of dimensions our problem has
    parameters = ["{0}".format(i) for i in range(mod_pars[0] * 5 + (mod_pars[0] - 1) * 6)]
    nparams = len(parameters)

    # make sure the output directories exist
    if not os.path.exists("./output/{0}/multinest".format(fname)):
        os.makedirs(os.path.join("./", "output", "{0}".format(fname), "multinest"))

    if not os.path.exists("./output/{0}/plots".format(fname)):
        os.makedirs(os.path.join("./", "output", "{0}".format(fname), "plots"))

    if not os.path.exists("chains"): os.makedirs("chains")
    # we want to see some output while it is running
    progress_plot = pymultinest.ProgressPlotter(n_params=nparams,
                                                outputfiles_basename='output/{0}/multinest/'.format(fname))
    progress_plot.start()
    # progress_print = pymultinest.ProgressPrinter(n_params=nparams, outputfiles_basename='output/{0}/multinest/'.format(fname))
    # progress_print.start()

    # run MultiNest
    pymultinest.run(lnlike, lnprior, nparams, outputfiles_basename=u'./output/{0}/multinest/'.format(fname),
                    resume=True, verbose=True,
                    sampling_efficiency='parameter', n_live_points=1000)

    # run has completed
    progress_plot.stop()
    # progress_print.stop()
    json.dump(parameters, open('./output/{0}/multinest/params.json'.format(fname), 'w'))  # save parameter names

    # plot the distribution of a posteriori possible models
    plt.figure()
    plt.plot(photo_data[0], photo_data[1], '+ ', color='red', label='data')

    a = pymultinest.Analyzer(outputfiles_basename="./output/{0}/reports/".format(fname), n_params=nparams)

    for theta in a.get_equal_weighted_posterior()[::100, :-1]:
        params = utilfuncs.split_parameters(theta, mod_pars[0])

        mod_flux, mod_rv = utilfuncs.model(mod_pars, params, photo_data[0], rv_data[0])

        plt.plot(photo_data[0], mod_flux, '-', color='blue', alpha=0.3, label='data')

    utilfuncs.report_as_input(params, fname)

    plt.savefig('./output/{0}/plots/posterior.pdf'.format(fname))
    plt.close()