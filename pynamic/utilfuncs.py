__author__ = 'nmearl'

import numpy as np
import pylab as pl
import triangle
import os
import photometry
import time


def model(params, flux_x, rv_x, ncores=1):
    x = np.append(flux_x, rv_x)
    x = x[np.argsort(x)]

    flux_inds = np.in1d(x, flux_x)
    rv_inds = np.in1d(x, rv_x)

    mod_flux, mod_rv = photometry.generate(params, x, ncores)

    return mod_flux[flux_inds], mod_rv[rv_inds]


def get_lmfit_parameters(params):
    N = params['N'].value
    t0 = params['t0'].value
    maxh = params['maxh'].value
    orbit_error = params['orbit_error'].value

    masses = np.array([params['mass_{0}'.format(i)].value for i in range(params['N'].value)])
    radii = np.array([params['radius_{0}'.format(i)].value for i in range(params['N'].value)])
    fluxes = np.array([params['flux_{0}'.format(i)].value for i in range(params['N'].value)])
    u1 = np.array([params['u1_{0}'.format(i)].value for i in range(params['N'].value)])
    u2 = np.array([params['u2_{0}'.format(i)].value for i in range(params['N'].value)])

    a = np.array([params['a_{0}'.format(i)].value for i in range(1, params['N'].value)])
    e = np.array([params['e_{0}'.format(i)].value for i in range(1, params['N'].value)])
    inc = np.array([params['inc_{0}'.format(i)].value for i in range(1, params['N'].value)])
    om = np.array([params['om_{0}'.format(i)].value for i in range(1, params['N'].value)])
    ln = np.array([params['ln_{0}'.format(i)].value for i in range(1, params['N'].value)])
    ma = np.array([params['ma_{0}'.format(i)].value for i in range(1, params['N'].value)])

    return N, t0, maxh, orbit_error, masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma


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


def split_parameters(params):
    N, t0, maxh, orbit_error = params[:4]
    N = int(N)

    body_params, kep_params = params[4:4 + (5 * N)], params[4 + (5 * N):]

    masses, radii, fluxes, u1, u2 = [np.array(body_params[i:N + i]) for i in range(0, len(body_params), N)]
    a, e, inc, om, ln, ma = [np.array(kep_params[i:N - 1 + i]) for i in range(0, len(kep_params), N - 1)]

    return N, t0, maxh, orbit_error, masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma


def reduced_chisqr(params, x, y, yerr, rv_data):
    N = params[0]

    mod_flux, mod_rv = model(params, x, rv_data[0])

    flnl = np.sum(((y - mod_flux) / yerr) ** 2) / (y.size - 1 - (N * 5 + (N - 1) * 6))
    rvlnl = np.sum(((rv_data[1] - mod_rv) / rv_data[2]) ** 2) / (rv_data[1].size - 1 - (N * 5 + (N - 1) * 6))

    return flnl #+ rvlnl


def iterprint(params, maxlnp, redchisqr, percomp, tleft):
    N, t0, maxh, orbit_error, \
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = params

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


def plot_out(params, fname, *args):
    N, t0, maxh, orbit_error, \
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = params

    print("Generating plots...")

    if not os.path.exists("./output"):
        os.mkdir("./output")

    if not os.path.exists("./output/{0}".format(fname)):
        os.mkdir("./output/{0}".format(fname))

    if not os.path.exists("./output/{0}/plots".format(fname)):
        os.mkdir("./output/{0}/plots".format(fname))

    names = ['mass', 'radii', 'flux', 'u1', 'u2', 'a', 'e', 'inc', 'om', 'ln', 'ma']


    if args:
        sampler, samples, ndim = args
        results = [masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma]

        # Plot corner plot
        # fig = triangle.corner(samples)
        # fig.savefig("./output/{0}/plots/triangle.png".format(fname))

        # Plot paths of walkers
        for i in range(len(results)):
            for j in range(len(results[i])):
                pl.clf()
                pl.plot(sampler.chain[:, :, i+j].T, color="k", alpha=0.4)
                pl.axhline(results[i][j], color="r", lw=2)
                pl.title("Walkers Distribution for {0:s}_{1:d}".format(names[i], j))
                pl.savefig('./output/{0}/plots/line_{1}_{2}.png'.format(fname, names[i], j))
                #pl.clf()

        # Plot value histograms
        for i in range(len(results)):
            for j in range(len(results[i])):
                pl.figure()
                pl.hist(sampler.flatchain[:, i+j], 100, color="k", histtype="step")
                pl.title("{0:s}_{1:d} Parameter Distribution".format(names[i], j))
                pl.savefig('./output/{0}/plots/dim_{1:s}_{2:d}.png'.format(fname, names[i], j))
                pl.close()


def mcmc_report_out(sys, results, fname):
    N, t0, maxh, orbit_error = sys
    N = int(N)

    GMsun = 2.959122083E-4 # AU**3/day**2
    Rsun = 0.00465116 # AU

    body_params, kep_params = results[:N*5], results[N*5:]

    masses, radii, fluxes, u1, u2 = [np.array(results[i:N+i]) for i in range(0, len(body_params), N)]
    a, e, inc, om, ln, ma = [np.array(results[i:N-1+i]) for i in range(0, len(kep_params), N-1)]

    params = [masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma]
    names = ['mass', 'radii', 'flux', 'u1', 'u2', 'a', 'e', 'inc', 'om', 'ln', 'ma']

    print('Saving results to file...')
    print(len(results))

    if not os.path.exists("./output"):
        os.mkdir("./output")

    if not os.path.exists("./output/{0}".format(fname)):
        os.mkdir("./output/{0}".format(fname))

    if not os.path.exists("./output/{0}/reports".format(fname)):
        os.mkdir("./output/{0}/reports".format(fname))

    with open('./output/{0}/reports/report.out'.format(fname), 'w') as f:

        print("="*50)
        print("Carter Units")
        print("-"*50)

        f.write("="*50)
        f.write("Carter Units")
        f.write("-"*50)

        for i in range(len(params)):
            param = params[i]
            name = names[i]

            for j in range(len(param)):
                print("{0}_{1} = {2[0]} +{2[1]} -{2[2]}".format(name, j, param[j]))
                f.write("{0}_{1} = {2[0]} +{2[1]} -{2[2]}".format(name, j, param[j]))

        print("="*50)
        print("Real Units")
        print("-"*50)

        f.write("="*50)
        f.write("Real Units")
        f.write("-"*50)

        for i in range(len(params)):
            param = params[i]
            name = names[i]

            if names[i] == 'mass':
                param /= GMsun

            elif names[i] == 'radii':
                param /= Rsun

            elif any(ang in names[i] for ang in ['inc', 'om', 'ln', 'ma']):
                param = np.rad2deg(param)

            for j in range(len(param)):
                print("{0}_{1} = {2[0]} +{2[1]} -{2[2]}".format(name, j, param[j]))
                f.write("{0}_{1} = {2[0]} +{2[1]} -{2[2]}".format(name, j, param[j]))


def report_as_input(params, fname):
    N, t0, maxh, orbit_error, \
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = params

    results = [masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma]

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