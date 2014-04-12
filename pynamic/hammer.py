from __future__ import print_function

import emcee
import numpy as np
import utilfuncs
import photometry
import os
import time
import minimizer


# Define the probability function as likelihood * prior.
def lnprior(theta, N):
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = utilfuncs.split_parameters(theta, N)

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
        # and np.all(a[1:] >= a[:-1]):
        return 0.0

    return -np.inf


def lnlike(theta, x, y, yerr, N, t0, maxh, orbit_error):
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = utilfuncs.split_parameters(theta, N)

    model = photometry.generate(
        N, t0, maxh, orbit_error,
        x,
        masses, radii, fluxes, u1, u2,
        a, e, inc, om, ln, ma
    )

    # lnf = np.log(1.0e-10)  # Natural log of the underestimation fraction
    # inv_sigma2 = 1.0 / (yerr ** 2 + model ** 2 * np.exp(2 * lnf))
    # return -0.5 * (np.sum((y - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))
    return (-0.5 * ((model - y) / yerr)**2).sum()


def lnprob(theta, x, y, yerr, N, t0, maxh, orbit_error):
    lp = lnprior(theta, N)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr, N, t0, maxh, orbit_error)


def generate(params, x, y, yerr, rv_data, nwalkers, niterations, ncores, randpars, fname):
    #np.seterr(all='raise')

    N, t0, maxh, orbit_error, masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = params
    theta = np.concatenate([masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma])
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

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr, N, t0, maxh, orbit_error), threads=ncores)

    # Clear and run the production chain.
    print("Running MCMC...")
    # pos, lnp, state = sampler.run_mcmc(pos0, 10, rstate0=np.random.get_state())
    # maxlnprob = np.argmax(lnp)
    # bestpos = pos[maxlnprob, :]
    #
    # redchisqr = utilfuncs.reduced_chisqr(bestpos, x, y, yerr, N, t0, maxh, orbit_error)
    #
    # utilfuncs.iterprint(N, bestpos, lnp[maxlnprob], redchisqr, 0.0 / niterations, 0.0)
    # utilfuncs.report_as_input(N, t0, maxh, orbit_error, utilfuncs.split_parameters(bestpos, N), fname)

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

        redchisqr = utilfuncs.reduced_chisqr(bestpos, x, y, yerr, N, t0, maxh, orbit_error)

        utilfuncs.iterprint(N, bestpos, lnp[maxlnprob], redchisqr, citer / niterations, tleft)
        utilfuncs.report_as_input(N, t0, maxh, orbit_error, utilfuncs.split_parameters(bestpos, N), fname)

    # Remove 'burn in' region
    print('Burning in; creating sampler chain...')

    burnin = int(.25 * niterations)
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

    utilfuncs.report_out(N, t0, maxh, orbit_error, results, fname)
    utilfuncs.plot_out(theta, fname, sampler, samples, ndim)