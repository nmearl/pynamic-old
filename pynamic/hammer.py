from __future__ import print_function

import emcee
import numpy as np
import utilfuncs
import modeler
import os
import time


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
        and len(inc[(inc > np.pi) | (inc < 0.0)]) == 0 \
        and len(om[(om > (2.0 * np.pi)) | (om < (-2.0 * np.pi))]) == 0 \
        and len(ln[(ln > np.pi) | (ln < -np.pi)]) == 0 \
        and len(ma[(ma > (2.0 * np.pi)) | (ma < 0.0)]) == 0 \
        and np.all(a[1:] >= a[:-1]):
        return 0.0
    return -np.inf


def lnlike(theta, x, y, yerr, N, t0, maxh, orbit_error):
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = utilfuncs.split_parameters(theta, N)

    model = modeler.generate(
        N, t0, maxh, orbit_error,
        x,
        masses, radii, fluxes, u1, u2,
        a, e, inc, om, ln, ma
    )

    # inv_sigma2 = 1.0 / (yerr ** 2 + model ** 2)# * np.exp(2.0))
    # return -0.5 * (np.sum((y - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))

    return -np.sum(((y - model) / yerr) ** 2) / (y.size - 1 - (N * 5 + (N - 1) * 6))


def lnprob(theta, x, y, yerr, N, t0, maxh, orbit_error):
    lp = lnprior(theta, N)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr, N, t0, maxh, orbit_error)


def generate(params, x, y, yerr, nwalkers, niterations, ncores, randpars, fname):
    #np.seterr(all='raise')

    N, t0, maxh, orbit_error, masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = params
    theta = np.concatenate((masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma))
    yerr = np.array(yerr)

    # print("Searching for maximum likelihood values...")

    # Find the maximum likelihood value.
    # chi2 = lambda *args: -2 * lnlike(*args)
    # result = op.minimize(
    #     chi2, theta,
    #     args=(x, y, yerr, N, t0, maxh, orbit_error)
    # )

    # Set up the sampler.
    ndim = len(theta)

    if randpars:
        pos0 = [np.concatenate(utilfuncs.random_pos(N)) for i in range(nwalkers)]
    else:
        # theta[theta == 0.0] += np.ones(len(theta))[theta == 0.0]
        pos0 = [theta + theta * 0.01 * np.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr, N, t0, maxh, orbit_error), threads=ncores)

    # Clear and run the production chain.
    print("Running MCMC...")
    # pos, prob, state = sampler.run_mcmc(pos, 100)#, rstate0=np.random.get_state())

    if not os.path.exists("./output"):
        os.mkdir("./output")

    f = open("output/chain_{0:s}.dat".format(fname), "w")
    f.close()

    # Setup some values for tracking time and completion
    citer, tlast, tsum = 0.0, time.time(), 0.0

    for pos, lnp, state in sampler.sample(pos0, iterations=niterations, storechain=True):
        citer += 1.0
        tsum += (time.time() - tlast)
        tleft = tsum / citer * (niterations - citer)
        tlast = time.time()

        maxlnprob = np.argmax(lnp)
        bestpos = pos[maxlnprob, :]

        iterprint(N, bestpos, lnp[maxlnprob], citer / niterations, tleft)
        utilfuncs.report_as_input(N, t0, maxh, orbit_error, utilfuncs.split_parameters(bestpos, N), fname)

    # Remove 'burn in' region
    print('Burning in; creating sampler chain...')

    burnin = 50
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


def iterprint(N, bestpos, maxlnp, percomp, tleft):
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = utilfuncs.split_parameters(bestpos, N)

    print('=' * 50)
    print('Probability: {0} | {1:2.1f}% complete, ~{2} left'.format(
        maxlnp, percomp * 100, time.strftime('%H:%M:%S', time.gmtime(tleft))))
    print('-' * 50)
    print('System parameters')
    print('-' * 50)
    print(
        '{0:11s} {1:11s} {2:11s} {3:11s} {4:11s} {5:11s} '.format(
            'Body', 'Mass', 'Radius', 'Flux', 'u1', 'u2'
        )
    )

    for i in range(N):
        print(
            '{0:11s} {1:1.5e} {2:1.5e} {3:1.5e} {4:1.5e} {5:1.5e}'.format(
                str(i + 1), masses[i], radii[i], fluxes[i], u1[i], u2[i]
            )
        )

    print('-' * 50)
    print('Keplerian parameters')
    print('-' * 50)

    print(
        '{0:11s} {1:11s} {2:11s} {3:11s} {4:11s} {5:11s} {6:11s}'.format(
            'Body', 'a', 'e', 'inc', 'om', 'ln', 'ma'
        )
    )

    for i in range(N - 1):
        print(
            '{0:11s} {1:1.5e} {2:1.5e} {3:1.5e} {4:1.5e} {5:1.5e} {6:1.5e}'.format(
                str(i + 2), a[i], e[i], inc[i], om[i], ln[i], ma[i]
            )
        )

    print()