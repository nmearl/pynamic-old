from __future__ import print_function

import emcee
import numpy as np
import utilfuncs
import photometry
import os
import time
import minimizer


# Define the probability function as likelihood * prior.
def lnprior(params):
    N, t0, maxh, orbit_error, \
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
        and len(ma[(ma > (2.0 * np.pi)) | (ma < 0.0)]) == 0:# \
        return 0.0

    return -np.inf


def lnlike(params, x, y, yerr, rv_data):

    # lnf = np.log(1.0e-10)  # Natural log of the underestimation fraction
    # inv_sigma2 = 1.0 / (yerr ** 2 + model ** 2 * np.exp(2 * lnf))
    # return -0.5 * (np.sum((y - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))

    if rv_data is not None:
        mod_flux, mod_rv = utilfuncs.model(params, x, rv_data[0])
        lnl = (-0.5 * ((mod_flux - y) / yerr)**2).sum()
        return lnl + (-0.5 * ((mod_rv - rv_data[1]) / rv_data[2])**2).sum()

    mod_flux, _ = utilfuncs.model(params, x)
    lnl = (-0.5 * ((mod_flux - y) / yerr)**2).sum()
    return lnl


def lnprob(theta, sys, x, y, yerr, rv_data):
    params = utilfuncs.split_parameters(np.append(sys, theta))
    lp = lnprior(params)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(params, x, y, yerr, rv_data)


def generate(params, x, y, yerr, rv_data, nwalkers, niterations, ncores, randpars, fname):
    # np.seterr(all='raise')

    N, t0, maxh, orbit_error, masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = params
    theta = np.concatenate([masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma])

    sys = [N, t0, maxh, orbit_error]

    yerr = np.array(yerr)

    # Set up the sampler.
    ndim = len(theta)

    # Generate parameters if no input file; give non zero amount to parameters with 0.0 value if given input file
    if randpars:
        print('Generating random parameters...')
        # pos0 = [np.concatenate(utilfuncs.random_pos(N)) for i in range(nwalkers)]
        pos0 = utilfuncs.random_pos(N, nwalkers)
    else:
        theta[theta == 0.0] = 1.0e-10
        pos0 = [theta + theta * 1.0e-3 * np.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(sys, x, y, yerr, rv_data), threads=ncores)

    # Clear and run the production chain.
    print("Running MCMC...")

    if not os.path.exists("./output"):
        os.mkdir("./output")

    # Setup some values for tracking time and completion
    citer, tlast, tsum = 0.0, time.time(), []

    for pos, lnp, state in sampler.sample(pos0, iterations=niterations, storechain=True):
        citer += 1.0
        tsum.append(time.time() - tlast)
        tleft = np.median(tsum) * (niterations - citer)
        tlast = time.time()

        maxlnprob = np.argmax(lnp)
        bestpos = pos[maxlnprob, :]

        params = utilfuncs.split_parameters(np.append(sys, bestpos))

        redchisqr = utilfuncs.reduced_chisqr(params, x, y, yerr)

        utilfuncs.iterprint(params, lnp[maxlnprob], redchisqr, citer / niterations, tleft)
        utilfuncs.report_as_input(params, fname)

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

    utilfuncs.mcmc_report_out(sys, results, fname)
    utilfuncs.plot_out(params, fname, sampler, samples, ndim)