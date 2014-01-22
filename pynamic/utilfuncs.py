__author__ = 'nmearl'

import numpy as np
import pylab as pl
import triangle
import os
import random


def random_pos(N):
    masses = [random.uniform(0.0, 0.01) for i in range(N)]
    radii = [random.uniform(0.0, 0.1) for i in range(N)]
    fluxes = [random.uniform(0.0, 1.0) for i in range(N)]
    u1 = [random.uniform(0.0, 1.0) for i in range(N)]
    u2 = [random.uniform(0.0, 1.0) for i in range(N)]
    a = [random.uniform(0.0, 1.0) for i in range(N - 1)]
    e = [random.uniform(0.0, 1.0) for i in range(N - 1)]
    inc = [random.uniform(0.0, np.pi) for i in range(N - 1)]
    om = [random.uniform(0.0, 2 * np.pi) for i in range(N - 1)]
    ln = [random.uniform(0.0, np.pi) for i in range(N - 1)]
    ma = [random.uniform(0.0, 2 * np.pi) for i in range(N - 1)]

    return masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma


def split_parameters(theta, N):
    theta = map(float, theta)
    sys_params, ind_params = theta[:5 * N], theta[5 * N:]

    masses, radii, fluxes, u1, u2 = [np.array(sys_params[i:N + i]) for i in range(0, len(sys_params), N)]
    a, e, inc, om, ln, ma = [np.array(ind_params[i:N - 1 + i]) for i in range(0, len(ind_params), N - 1)]

    # print_out([masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma], N)
    # for par in [masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma]:
    #     print(par)

    return masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma


def plot_out(theta, fname, *args):
    print("Generating plots...")

    if not os.path.exists("./output"):
        os.mkdir("./output")

    if not os.path.exists("./output/plots"):
        os.mkdir("./output/plots")

    if args:
        sampler, samples, ndim = args

        # Plot corner plot
        fig = triangle.corner(samples)
        fig.savefig("./output/plots/triangle_{0}.eps".format(fname))

        # Plot paths of walkers
        for i in range(len(theta)):
            pl.clf()
            pl.plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
            pl.axhline(theta[i], color="r", lw=2)
            pl.savefig('./output/plots/line_{0}_{1}.png'.format(i, fname))
            #pl.clf()

        # Plot value histograms
        for i in range(ndim):
            pl.figure()
            pl.hist(sampler.flatchain[:, i], 100, color="k", histtype="step")
            pl.title("Dimension {0:d}".format(i))
            pl.savefig('./output/plots/dim_{0:d}.eps'.format(i))
            pl.close()


def report_out(results, N, fname):
    GMsun = 2.959122083E-4 # AU**3/day**2
    Rsun = 0.00465116 # AU

    results = split_parameters(results, N)
    names = ['mass', 'radii', 'flux', 'u1', 'u2', 'a', 'e', 'inc', 'om', 'ln', 'ma']

    print('Saving results to file...')

    if not os.path.exists("./output"):
        os.mkdir("./output")

    if not os.path.exists("./output/reports"):
        os.mkdir("./output/reports")

    with open('./output/reports/report_{0}.out'.format(fname), 'w') as f:
        for i in range(len(results)):
            param = np.array(results[i])

            for j in range(len(param)):

                print("{0:8s}_{1} = {2[0]:20.11e} +{2[1]:20.11e} -{2[2]:20.11e}".format(names[i], j, param[j]))
                f.write("{0}_{1} = {2[0]} +{2[1]} -{2[2]}\n".format(names[i], j, param[j]))

                if names[i] == 'mass':
                    param /= GMsun

                elif names[i] == 'radii':
                    param /= Rsun

                elif any(ang in names[i] for ang in ['inc', 'om', 'ln', 'ma']):
                    param = np.rad2deg(param)

                print("{0}_{1} = {2[0]} +{2[1]} -{2[2]}\n".format(names[i], j, param[j]))
                f.write("{0}_{1} = {2[0]} +{2[1]} -{2[2]}\n".format(names[i], j, param[j]))


def report_as_input(results, N, fname):
    results = split_parameters(results, N)

    if not os.path.exists("./output"):
        os.mkdir("./output")

    if not os.path.exists("./output/reports"):
        os.mkdir("./output/reports")

    with open('./output/reports/input_out_{0}.out'.format(fname), 'w') as f:
        for i in range(len(results)):
            param = results[i]

            f.write('{0:s}\n'.format(' '.join(map(str, param))))