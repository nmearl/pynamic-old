#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import emcee
import triangle
import numpy as np
import scipy.optimize as op
import pylab as pl
from matplotlib.ticker import MaxNLocator
import modeler


def _split_parameters(theta, N):
    sys_params, ind_params = theta[:5 * N], theta[5 * N:]

    masses, radii, fluxes, u1, u2 = [np.array(sys_params[i:N + i]) for i in range(0, len(sys_params), N)]
    a, e, inc, om, ln, ma = [np.array(ind_params[i:N - 1 + i]) for i in range(0, len(ind_params), N - 1)]

    # print_out([masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma], N)
    # for par in [masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma]:
    #     print(par)

    return masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma


# Define the probability function as likelihood * prior.
def lnprior(theta, N):
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = _split_parameters(theta, N)

    # print('mas', masses, len(masses[masses <= 0.0]) == 0)
    # print('rad', radii, len(radii[radii <= 0.0]) == 0)
    # print('u1', u1, len(u1[(u1 > 1.0) | (u1 < 0.0)]) == 0)
    # print('u2', u2, len(u2[(u2 > 1.0) | (u2 < 0.0)]) == 0)
    # print('a', a, len(a[a < 0.0]) == 0)
    # print('e', e, len(e[(e > 1.0) | (e < 0.0)]) == 0)
    # print('inc', inc, len(inc[(inc > (2.0 * np.pi)) | (inc < (-2.0 * np.pi))]) == 0)
    # print('om', om, len(om[(om > (2.0 * np.pi)) | (om < (-2.0 * np.pi))]) == 0)
    # print('ln', ln, len(ln[(ln > (2.0 * np.pi)) | (ln < (-2.0 * np.pi))]) == 0)
    # print('ma', ma, len(ma[(ma > (2.0 * np.pi)) | (ma < (-2.0 * np.pi))]) == 0)

    if len(masses[(masses <= 0.0) | (masses > 0.01)]) == 0 \
        and len(radii[(radii <= 0.0) | (radii > 0.1)]) == 0 \
        and len(fluxes[(fluxes > 1.0) | (fluxes < 0.0)]) == 0 \
        and len(u1[(u1 > 1.0) | (u1 < 0.0)]) == 0 \
        and len(u2[(u2 > 1.0) | (u2 < 0.0)]) == 0 \
        and len(a[a < 0.0]) == 0 \
        and len(e[(e > 1.0) | (e < 0.0)]) == 0 \
        and len(inc[(inc > (2.0 * np.pi)) | (inc < 0.0)]) == 0 \
        and len(om[(om > (2.0 * np.pi)) | (om < 0.0)]) == 0 \
        and len(ln[(ln > np.pi) | (ln < 0.0)]) == 0 \
        and len(ma[(ma > (2.0 * np.pi)) | (ma < 0.0)]) == 0 \
        and (masses[2] > masses[0] > masses[1]):

        print('\nComparing prior function...')
        print('{0:11s} {1:11s} {2:11s} {3:11s} {4:11s} {5:11s} {6:11s} {7:11s} {8:11s} {9:11s} {10:11s}'.format(
            'Mass', 'Radius', 'Flux', 'u1', 'u2', 'a', 'e', 'inc', 'om', 'ln', 'ma'
        ))

        for i in range(N):
            if i == 0:
                print('{0:1.5e} {1:1.5e} {2:1.5e} {3:1.5e} {4:1.5e}'.format(
                    masses[i], radii[i], fluxes[i], u1[i], u2[i]
                ))

            else:
                print(
                    '{0:1.5e} {1:1.5e} {2:1.5e} {3:1.5e} {4:1.5e} {5:1.5e} {6:1.5e} {7:1.5e} {8:1.5e} {9:1.5e} {10:1.5e}'.format(
                        masses[i], radii[i], fluxes[i], u1[i], u2[i], a[i - 1], e[i - 1], inc[i - 1], om[i - 1],
                        ln[i - 1], ma[i - 1]
                    ))

        return 0.0
    return -np.inf


def lnlike(theta, x, y, yerr, N, t0, maxh, orbit_error):
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = _split_parameters(theta, N)

    print('Creating likelihood function...')
    model = modeler.generate(
        N, t0, maxh, orbit_error,
        x,
        masses, radii, fluxes, u1, u2,
        a, e, inc, om, ln, ma
    )
    print('Done.')

    inv_sigma2 = 1.0 / (yerr ** 2 + model ** 2)# * np.exp(2.0))
    return -0.5 * (np.sum((y - model) ** 2 * inv_sigma2 - np.log(inv_sigma2)))


def lnprob(theta, x, y, yerr, N, t0, maxh, orbit_error):
    lp = lnprior(theta, N)
    if not np.isfinite(lp):
        return -np.inf
    print('Found viable probability function...')
    return lp + lnlike(theta, x, y, yerr, N, t0, maxh, orbit_error)


def generate(params, x, y, yerr):
    #np.seterr(all='raise')

    N, t0, maxh, orbit_error, masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = params
    theta = np.array(masses + radii + fluxes + u1 + u2 + a + e + inc + om + ln + ma)
    yerr = np.array(yerr)

    print("Searching for maximum likelihood values...")

    # Find the maximum likelihood value.
    # chi2 = lambda *args: -2 * lnlike(*args)
    # result = op.minimize(
    #     chi2, theta,
    #     args=(x, y, yerr, N, t0, maxh, orbit_error)
    # )

    print(theta)


    #print("""Maximum likelihood result:
    #    m = {0} (truth: {1})
    #    b = {2} (truth: {3})
    #    f = {4} (truth: {5})
    #""".format(m_ml, m_true, b_ml, b_true, np.exp(lnf_ml), f_true))

    # Set up the sampler.
    ndim, nwalkers = len(theta), 54
    pos = [theta + theta * 0.1 * np.random.randn(ndim)
           # if not theta == 0.0 else theta + 0.1 * np.random.randn(ndim)
           for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr, N, t0, maxh, orbit_error))#, threads=2)

    # Clear and run the production chain.
    print("Running MCMC...", )
    # pos, prob, state = sampler.run_mcmc(pos, 100)#, rstate0=np.random.get_state())
    f = open("chain.dat", "w")
    f.close()

    for result in sampler.sample(pos, iterations=250, storechain=False):
        position, lnprobability = result[0], result[1]

        with open("chain.dat", "a") as f:
            for k in range(position.shape[0]):
                f.write("{0:4d} {1:s} {2:s}\n".format(k, str(lnprobability[k]), " ".join(map(str, position[k]))))

    print("Done.")
    print("Final parameter set: ")
    print(pos.shape)

    for i in range(len(theta)):
        pl.clf()
        pl.plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
        pl.axhline(theta[i], color="r", lw=2)
        pl.savefig('./output/line_{0}.png'.format(i))
        #pl.clf()
    #fig, axes = pl.subplots(3, 1, sharex=True, figsize=(8, 9))
    #axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
    #axes[0].yaxis.set_major_locator(MaxNLocator(5))
    #axes[0].axhline(m_true, color="#888888", lw=2)
    #axes[0].set_ylabel("$m$")
    #
    #axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
    #axes[1].yaxis.set_major_locator(MaxNLocator(5))
    #axes[1].axhline(b_true, color="#888888", lw=2)
    #axes[1].set_ylabel("$b$")
    #
    #axes[2].plot(np.exp(sampler.chain[:, :, 2]).T, color="k", alpha=0.4)
    #axes[2].yaxis.set_major_locator(MaxNLocator(5))
    #axes[2].axhline(f_true, color="#888888", lw=2)
    #axes[2].set_ylabel("$f$")
    #axes[2].set_xlabel("step number")
    #
    #fig.tight_layout(h_pad=0.0)
    #fig.savefig("line-time.png")

    # Make the triangle plot.
    print('Burning in; creating sampler chain...')
    burnin = 1
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

    print(samples.shape)
    print(samples[:, 0])

    fig = triangle.corner(samples)
    fig.savefig("./output/kep47-triangle.eps")

    # Compute the quantiles.
    print('Computing quantiles...')
    # samples[:, len(theta) - 1] = np.exp(samples[:, len(theta) - 1])

    print('Mapping results...')
    results = np.array(map(
        lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
        zip(*np.percentile(samples, [16, 50, 84],
                           axis=0))
    ))

    report_out(results, N)

    print('Modelling final results...')
    # Produce final model and save the values
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = _split_parameters(results[:, 0], N)

    model = modeler.generate(
        N, t0, maxh, orbit_error,
        x,
        masses, radii, fluxes, u1, u2,
        a, e, inc, om, ln, ma
    )

    np.savez('./output/mcmc_kep126', x, y, model)

    plotter(sampler, samples, ndim, x, y, model)


def report_out(results, N):
    GMsun = 2.959122083E-4 # AU**3/day**2
    Rsun = 0.00465116 # AU

    results = _split_parameters(results, N)
    names = ['mass', 'radii', 'flux', 'u1', 'u2', 'a', 'e', 'inc', 'om', 'ln', 'ma']

    print('Saving results to file...')

    with open('./output/report_mcmc.out', 'w') as f:
        for i in range(len(results)):
            param = np.array(results[i])

            for j in range(len(param)):

                print("{0:8s}_{1} = {2[0]:20.11e} +{2[1]:20.11e} -{2[2]:20.11e}".format(names[i], j, param[j]))

                if names[i] == 'mass':
                    param /= GMsun

                elif names[i] == 'radii':
                    param /= Rsun

                elif any(ang in names[i] for ang in ['inc', 'om', 'ln', 'ma']):
                    param = np.rad2deg(param)

                print("{0}_{1} = {2[0]} +{2[1]} -{2[2]}\n".format(names[i], j, param[j]))
                f.write("{0}_{1} = {2[0]} +{2[1]} -{2[2]}\n".format(names[i], j, param[j]))


def print_out(results, N):
    names = ['mass', 'radii', 'flux', 'u1', 'u2', 'a', 'e', 'inc', 'om', 'ln', 'ma']

    for i in range(len(results)):
        param = results[i]

        for j in range(len(param)):
            # if names[i] == 'mass':
            #     param /= GMsun
            #
            # elif names[i] == 'radius':
            #     param /= Rsun
            #
            # elif any(ang in names[i] for ang in ['inc', 'om', 'ln', 'ma']):
            #     param = np.rad2deg(param)

            print("{0:8s}_{1} = {2[0]:20.11e} +{2[1]:20.11e} -{2[2]:20.11e}\n".format(names[i], j, param[j]))


def plotter(sampler, samples, ndim, x, y, model):
    print("Generating plots...")
    #N, t0, maxh, orbit_error = sys_params
    #masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = _split_parameters(results, N)
    #
    # Plot value histograms
    # for i in range(ndim):
    #     pl.figure()
    #     pl.hist(sampler.flatchain[:, i], 100, color="k", histtype="step")
    #     pl.title("Dimension {0:d}".format(i))
    #     pl.savefig('./output/plots/dim_{0:d}.eps'.format(i))
    #     pl.close()

    pl.plot(x, model, 'k+')
    pl.plot(x, y, 'r')
    pl.savefig('./output/mcmc_kep126.png')