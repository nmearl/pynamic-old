__author__ = 'nmearl'

import numpy as np
import pylab as pl
import triangle
import os
import photometry
import time


def random_pos(N, nwalkers):
    all_masses = 10**-np.logspace(0.0, 1.0, nwalkers)
    all_radii = 10 * 10**-np.logspace(0.4, 0.7, nwalkers)
    all_fluxes = np.random.uniform(0.0, 1.0, nwalkers)
    all_u1 = np.random.uniform(0.0, 1.0, nwalkers)
    all_u2 = np.random.uniform(0.0, 1.0, nwalkers)
    all_a = 10 * 10**-np.logspace(0.0, 0.6, nwalkers)
    all_e = np.random.uniform(0.0, 1.0, nwalkers)
    all_inc = np.random.normal(np.pi/2, 0.1, nwalkers)
    all_om = np.random.uniform(-2 * np.pi, 2 * np.pi, nwalkers)
    all_ln = np.random.uniform(-np.pi, np.pi, nwalkers)
    all_ma = np.random.uniform(0.0, 2 * np.pi, nwalkers)

    pos = []
    for i in range(nwalkers):
        masses = np.array([all_masses[i] for i in np.random.randint(nwalkers, size=N)])
        radii = np.array([all_radii[i] for i in np.random.randint(nwalkers, size=N)])
        fluxes = np.array([all_fluxes[i] for i in np.random.randint(nwalkers, size=N)])
        u1 = np.array([all_u1[i] for i in np.random.randint(nwalkers, size=N)])
        u2 = np.array([all_u2[i] for i in np.random.randint(nwalkers, size=N)])
        a = np.sort(np.array([all_a[i] for i in np.random.randint(nwalkers, size=N-1)]))
        e = np.array([all_e[i] for i in np.random.randint(nwalkers, size=N-1)])
        inc = np.array([all_inc[i] for i in np.random.randint(nwalkers, size=N-1)])
        om = np.array([all_om[i] for i in np.random.randint(nwalkers, size=N-1)])
        ln = np.array([all_ln[i] for i in np.random.randint(nwalkers, size=N-1)])
        ma = np.array([all_ma[i] for i in np.random.randint(nwalkers, size=N-1)])

        pos.append(np.concatenate([masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma]))

    return np.array(pos)


def find_nearest(array, targets):
    res = []

    for tar in targets:
        idx = np.argmin(np.abs(array - tar))
        res.append(idx)

    return np.array(res)


def split_parameters(theta, N):
    sys_params, ind_params = theta[:5 * N], theta[5 * N:]

    masses, radii, fluxes, u1, u2 = [np.array(sys_params[i:N + i]) for i in range(0, len(sys_params), N)]
    a, e, inc, om, ln, ma = [np.array(ind_params[i:N - 1 + i]) for i in range(0, len(ind_params), N - 1)]

    return masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma


def reduced_chisqr(theta, x, y, yerr, N, t0, maxh, orbit_error):
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = split_parameters(theta, N)

    mod_flux, mod_rv = photometry.generate(
        N, t0, maxh, orbit_error,
        x,
        masses, radii, fluxes, u1, u2,
        a, e, inc, om, ln, ma
    )

    return np.sum(((y - mod_flux) / yerr) ** 2) / (y.size - 1 - (N * 5 + (N - 1) * 6))


def iterprint(N, bestpos, maxlnp, redchisqr, percomp, tleft):
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = split_parameters(bestpos, N)

    print('=' * 80)
    print('Likelihood: {0}, Red. Chi: {1} | {2:2.1f}% complete, ~{3} left'.format(
        maxlnp, redchisqr, percomp * 100, time.strftime('%H:%M:%S', time.gmtime(tleft))))
    print('-' * 80)
    print('System parameters')
    print('-' * 80)
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

    print('-' * 80)
    print('Keplerian parameters')
    print('-' * 80)

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

    print('')


def plot_out(theta, fname, *args):
    print("Generating plots...")

    if not os.path.exists("./output"):
        os.mkdir("./output")

    if not os.path.exists("./output/{0}".format(fname)):
        os.mkdir("./output/{0}".format(fname))

    if not os.path.exists("./output/{0}/plots".format(fname)):
        os.mkdir("./output/{0}/plots".format(fname))

    if args:
        sampler, samples, ndim = args

        # Plot corner plot
        # fig = triangle.corner(samples)
        # fig.savefig("./output/{0}/plots/triangle.png".format(fname))

        # Plot paths of walkers
        for i in range(len(theta)):
            pl.clf()
            pl.plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
            pl.axhline(theta[i], color="r", lw=2)
            pl.savefig('./output/{0}/plots/line_{1}.png'.format(fname, i))
            #pl.clf()

        # Plot value histograms
        for i in range(ndim):
            pl.figure()
            pl.hist(sampler.flatchain[:, i], 100, color="k", histtype="step")
            pl.title("Dimension {0:d}".format(i))
            pl.savefig('./output/{0}/plots/dim_{1:d}.png'.format(fname, i))
            pl.close()


def report_out(N, t0, maxh, orbit_error, results, fname):
    results = np.array(results)
    report_as_input(N, t0, maxh, orbit_error, split_parameters(results[:, 0], N), fname)

    results = split_parameters(results, N)

    GMsun = 2.959122083E-4 # AU**3/day**2
    Rsun = 0.00465116 # AU

    names = ['mass', 'radii', 'flux', 'u1', 'u2', 'a', 'e', 'inc', 'om', 'ln', 'ma']

    print('Saving results to file...')

    if not os.path.exists("./output"):
        os.mkdir("./output")

    if not os.path.exists("./output/{0}".format(fname)):
        os.mkdir("./output/{0}".format(fname))

    if not os.path.exists("./output/{0}/reports".format(fname)):
        os.mkdir("./output/{0}/reports".format(fname))

    with open('./output/{0}/reports/report.out'.format(fname), 'w') as f:
        for i in range(len(results)):
            param = results[i]

            for j in range(len(param)):
                print("{0}_{1} = {2[0]} +{2[1]} -{2[2]}".format(names[i], j, param[j]))
                f.write("{0}_{1} = {2[0]} +{2[1]} -{2[2]}".format(names[i], j, param[j]))

            if names[i] == 'mass':
                param /= GMsun

            elif names[i] == 'radii':
                param /= Rsun

            elif any(ang in names[i] for ang in ['inc', 'om', 'ln', 'ma']):
                param = np.rad2deg(param)

            print('')

            for j in range(len(param)):
                print("{0}_{1} = {2[0]} +{2[1]} -{2[2]}".format(names[i], j, param[j]))
                f.write("{0}_{1} = {2[0]} +{2[1]} -{2[2]}".format(names[i], j, param[j]))

            print('')


def report_as_input(N, t0, maxh, orbit_error, results, fname):
    if not os.path.exists("./output"):
        os.mkdir("./output")

    if not os.path.exists("./output/{0}".format(fname)):
        os.mkdir("./output/{0}".format(fname))
        os.mkdir("./output/{0}/reports".format(fname))

    with open('./output/{0}/reports/input_final_{1}.out'.format(fname, fname), 'w') as f:
        f.write("{0:d}\n".format(N))
        f.write("{0:f}\n".format(t0))
        f.write("{0:f}\n".format(maxh))
        f.write("{0:e}\n".format(orbit_error))

        for i in range(len(results)):
            param = results[i]

            f.write('{0:s}\n'.format(' '.join(map(str, param))))