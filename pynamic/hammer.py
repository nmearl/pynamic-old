from __future__ import print_function

import emcee
import numpy as np
import utilfuncs
import modeler


# Define the probability function as likelihood * prior.
def lnprior(theta, N):
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = utilfuncs.split_parameters(theta, N)

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
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = utilfuncs.split_parameters(theta, N)

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


def generate(params, x, y, yerr, fname):
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

    # Set up the sampler.
    ndim, nwalkers = len(theta), 250
    pos = [theta + theta * 0.1 * np.random.randn(ndim)
           # if not theta == 0.0 else theta + 0.1 * np.random.randn(ndim)
           for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr, N, t0, maxh, orbit_error))#, threads=2)

    # Clear and run the production chain.
    print("Running MCMC...", )
    # pos, prob, state = sampler.run_mcmc(pos, 100)#, rstate0=np.random.get_state())
    f = open("./output/chain_{0:s}.dat".format(fname), "w")
    f.close()

    for ipos, lnp, state in sampler.sample(pos, iterations=1000, storechain=False):
        maxlnprob = np.argmax(lnp)
        bestipos = ipos[maxlnprob, :]

        with open("./output/chain_{0:s}.dat".format(fname), "a") as f:
            f.write("{0:s} {1:s}\n".format(str(maxlnprob), " ".join(map(str, bestipos))))

    print("Done.")

    print('Burning in; creating sampler chain...')
    burnin = 50
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

    # Compute the quantiles.
    print('Computing quantiles; mapping results...')
    results = np.array(
        map(
            lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
            zip(*np.percentile(samples, [16, 50, 84],
                               axis=0))
        )
    )

    print('Modelling final results...')
    # Produce final model and save the values
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = utilfuncs.split_parameters(results[:, 0], N)

    model = modeler.generate(
        N, t0, maxh, orbit_error,
        x,
        masses, radii, fluxes, u1, u2,
        a, e, inc, om, ln, ma
    )

    np.savez('./output/mcmc_{0}'.format(fname), x, y, model)

    utilfuncs.report_out(results, N, fname)
    utilfuncs.plot_out(x, y, model, theta, fname, sampler, samples, ndim)