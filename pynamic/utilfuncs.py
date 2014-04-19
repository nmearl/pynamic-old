__author__ = 'nmearl'

import numpy as np
import pylab as pl
import triangle
import os
import photometry
import time


def model(mod_pars, params, flux_x, rv_x, ncores=1):
    x = np.append(flux_x, rv_x)
    x = np.unique(x[np.argsort(x)])

    flux_inds = np.in1d(x, flux_x, assume_unique=True)
    rv_inds = np.in1d(x, rv_x, assume_unique=True)

    mod_flux, mod_rv = photometry.generate(mod_pars, params, x, ncores)

    return mod_flux[flux_inds], mod_rv[rv_inds]


def get_lmfit_parameters(mod_pars, params):
    N, t0, maxh, orbit_error, rv_body, rv_corr = mod_pars

    masses = np.array([params['mass_{0}'.format(i)].value for i in range(N)])
    radii = np.array([params['radius_{0}'.format(i)].value for i in range(N)])
    fluxes = np.array([params['flux_{0}'.format(i)].value for i in range(N)])
    u1 = np.array([params['u1_{0}'.format(i)].value for i in range(N)])
    u2 = np.array([params['u2_{0}'.format(i)].value for i in range(N)])

    a = np.array([params['a_{0}'.format(i)].value for i in range(1, N)])
    e = np.array([params['e_{0}'.format(i)].value for i in range(1, N)])
    inc = np.array([params['inc_{0}'.format(i)].value for i in range(1, N)])
    om = np.array([params['om_{0}'.format(i)].value for i in range(1, N)])
    ln = np.array([params['ln_{0}'.format(i)].value for i in range(1, N)])
    ma = np.array([params['ma_{0}'.format(i)].value for i in range(1, N)])

    return masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma


def split_parameters(params, nbodies):
    body_params, kep_params = params[:(5 * nbodies)], params[(5 * nbodies):]

    masses, radii, fluxes, u1, u2 = [np.array(body_params[i:nbodies + i]) for i in range(0, len(body_params), nbodies)]
    a, e, inc, om, ln, ma = [np.array(kep_params[i:nbodies - 1 + i]) for i in range(0, len(kep_params), nbodies - 1)]

    return masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma


def reduced_chisqr(mod_pars, params, photo_data, rv_data):
    mod_flux, mod_rv = model(mod_pars, params, photo_data[0], rv_data[0])

    nbodies = mod_pars[0]

    flnl = np.sum(((photo_data[1] - mod_flux) / photo_data[2]) ** 2) / \
           (photo_data[1].size - 1 - (nbodies * 5 + (nbodies - 1) * 6))
    rvlnl = np.sum(((rv_data[1] - mod_rv) / rv_data[2]) ** 2) / \
            (rv_data[1].size - 1 - (nbodies * 5 + (nbodies - 1) * 6))

    return flnl #+ rvlnl


def iterprint(mod_pars, params, maxlnp, redchisqr, percomp, tleft):
    nbodies = mod_pars[0]
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = params

    print('=' * 83)
    print('Likelihood: {0}, Red. Chi: {1} | {2:2.1f}% complete, ~{3} left'.format(
        maxlnp, redchisqr, percomp * 100, time.strftime('%H:%M:%S', time.gmtime(tleft))))
    print('-' * 83)
    print('System parameters')
    print('-' * 83)
    print(
        '{0:11s} {1:11s} {2:11s} {3:11s} {4:11s} {5:11s} '.format(
            'Body', 'Mass', 'Radius', 'Flux', 'u1', 'u2'
        )
    )

    for i in range(nbodies):
        print(
            '{0:11s} {1:1.5e} {2:1.5e} {3:1.5e} {4:1.5e} {5:1.5e}'.format(
                str(i + 1), masses[i], radii[i], fluxes[i], u1[i], u2[i]
            )
        )

    print('-' * 83)
    print('Keplerian parameters')
    print('-' * 83)

    print(
        '{0:11s} {1:11s} {2:11s} {3:11s} {4:11s} {5:11s} {6:11s}'.format(
            'Body', 'a', 'e', 'inc', 'om', 'ln', 'ma'
        )
    )

    for i in range(nbodies - 1):
        print(
            '{0:11s} {1:1.5e} {2:1.5e} {3:1.5e} {4:1.5e} {5:1.5e} {6:1.5e}'.format(
                str(i + 2), a[i], e[i], inc[i], om[i], ln[i], ma[i]
            )
        )

    print('')


def plot_out(params, fname, *args):
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


def mcmc_report_out(mod_pars, results, fname):
    N, t0, maxh, orbit_error, rv_body, rv_corr = mod_pars
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


def report_as_input(mod_pars, params, fname):
    N, t0, maxh, orbit_error, rv_body, rv_corr = mod_pars

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

        for i in range(len(params)):
            param = params[i]

            f.write('{0:s}\n'.format(' '.join(map(str, param))))