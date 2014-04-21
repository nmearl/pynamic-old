from __future__ import print_function

import emcee
import numpy as np
import utilfuncs
import photometry
import os
import time
import minimizer
import itertools


# Define the probability function as likelihood * prior.
def lnprior(params):
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = params

    if len(masses[(masses <= 0.0) | (masses > 0.1)]) == 0 \
        and len(radii[(radii <= 0.0) | (radii > 1.0)]) == 0 \
        and len(fluxes[(fluxes > 1.0) | (fluxes < 0.0)]) == 0 \
        and len(u1[(u1 > 1.0) | (u1 < 0.0)]) == 0 \
        and len(u2[(u2 > 1.0) | (u2 < 0.0)]) == 0 \
        and len(a[(a < 0.0) | (a > 100.0)]) == 0 \
        and len(e[(e > 1.0) | (e < 0.0)]) == 0 \
        and len(inc[(inc > 2.0 * np.pi) | (inc < 0.0)]) == 0 \
        and len(om[(om > (2.0 * np.pi)) | (om < -(2.0 * np.pi))]) == 0 \
        and len(ln[(ln > (2.0 * np.pi)) | (ln < -(2.0 * np.pi))]) == 0 \
        and len(ma[(ma > (2.0 * np.pi)) | (ma < 0.0)]) == 0:  # \
        return 0.0

    return -np.inf


def lnlike(mod_pars, params, photo_data, rv_data):
    # lnf = np.log(1.0e-10)  # Natural log of the underestimation fraction
    # inv_sigma2 = 1.0 / (yerr ** 2 + model ** 2 * np.exp(2 * lnf))
    # return -0.5 * (np.sum((y - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))

    mod_flux, mod_rv = utilfuncs.model(mod_pars, params, photo_data[0], rv_data[0])
    flnl = np.sum((-0.5 * ((mod_flux - photo_data[1]) / photo_data[2]) ** 2))
    rvlnl = np.sum((-0.5 * ((mod_rv - rv_data[1]) / rv_data[2])**2))

    return flnl + rvlnl

    # mod_flux, _ = utilfuncs.model(params, x)
    # lnl = (-0.5 * ((mod_flux - y) / yerr)**2).sum()
    # return lnl


def lnprob(theta, mod_pars, photo_data, rv_data):
    params = utilfuncs.split_parameters(theta, mod_pars[0])
    lp = lnprior(params)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(mod_pars, params, photo_data, rv_data)


def generate(mod_pars, body_pars, photo_data, rv_data, nwalkers, ncores, fname, niterations=500):
    # Flatten body parameters
    theta = np.array(list(itertools.chain.from_iterable(body_pars)))

    # Set up the sampler.
    ndim = len(theta)
    theta[theta == 0.0] = 1.0e-10
    pos0 = [theta + theta * 1.0e-3 * np.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(mod_pars, photo_data, rv_data), threads=ncores)

    # Clear and run the production chain.
    print("Running MCMC...")

    # Make sure paths exist
    if not os.path.exists("./output/{0}/reports".format(fname)):
        os.makedirs(os.path.join("./", "{0}".format(fname), "output", "reports"))

    # Setup some values for tracking time and completion
    citer, tlast, tsum = 0.0, time.time(), []

    for pos, lnp, state in sampler.sample(pos0, iterations=niterations, storechain=True):
        # Save out the chain for later analysis
        with open("./output/{0}/reports/mcmc_chain.dat".format(fname), "a+") as f:
            for k in range(pos.shape[0]):
                f.write("{0:4d} {1:s}\n".format(k, " ".join(map(str, pos[k]))))

        citer += 1.0
        tsum.append(time.time() - tlast)
        tleft = np.median(tsum) * (niterations - citer)
        tlast = time.time()

        maxlnprob = np.argmax(lnp)
        bestpos = pos[maxlnprob, :]

        params = utilfuncs.split_parameters(bestpos, mod_pars[0])

        redchisqr = utilfuncs.reduced_chisqr(mod_pars, params, photo_data, rv_data)

        utilfuncs.iterprint(mod_pars, params, lnp[maxlnprob], redchisqr, citer / niterations, tleft)
        utilfuncs.report_as_input(mod_pars, params, fname)

    # Remove 'burn in' region
    print('Burning in; creating sampler chain...')

    burnin = int(0.5 * niterations)
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

    # Compute the quantiles.
    print('Computing quantiles; mapping results...')

    results = map(
        lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
        zip(*np.percentile(samples, [16, 50, 84],
                           axis=0))
    )

    # Produce final model and save the values
    print('Saving final results...')

    utilfuncs.mcmc_report_out(mod_pars, results, fname)
    utilfuncs.plot_out(params, fname, sampler, samples, ndim)