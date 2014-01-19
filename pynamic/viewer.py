__author__ = 'nmearl'

import numpy as np
import pylab
from matplotlib import gridspec
import utilfuncs
import modeler


def viewnp(filename, show_residuals=False):
    data = np.load(filename)
    x, y, model = data['arr_0'], data['arr_1'], data['arr_2']

    if show_residuals:
        fig = pylab.figure()
        gs = gridspec.GridSpec(1, 1, height_ratios=[1, 0])

        ax1 = pylab.subplot(gs[0])
        # ax1.errorbar(x, y, yerr=yerr, fmt='o', label='Data')
        ax1.plot(x, y, 'k+', label='Data')
        # ax1.plot(x, data + residuals, 'b')
        ax1.plot(x, model, 'r', label="Model")

        pylab.legend(loc=0)
        pylab.xlabel('Time (BJD)')
        pylab.ylabel('Normalized Flux')

        #ax2 = pylab.subplot(gs[1])
        #ax2.plot(x, residuals)
        # pylab.savefig('./out_plot.eps')
        pylab.show()


def plotter(filename):
    par_dict = {}

    with open(filename, 'r') as f:
        hl = False
        for line in f:
            nline = line.strip().split()

            if not hl:
                hline = nline
                hl = True
                continue

            for i in range(len(hline)):
                if not hline[i] in par_dict.keys():
                    par_dict[hline[i]] = [float(nline[i]), ]
                else:
                    par_dict[hline[i]].append(float(nline[i]))

    # pylab.fylim(0.1e7, 1e7)
    pylab.semilogy(par_dict['radius_0'], par_dict['chi_squared'], 'r+')
    pylab.show()


def chain_plotter(filename):
    n, lnp, params = [], [], []

    with open(filename, 'r') as f:
        print(len(f.readlines()))
        for line in f:
            line = line.strip().split()

            lnp.append(float(line[0]))
            params.append(line[1:])

    for i in range(len(params)):
        params[i] = map(float, params[i])

    chain_list = zip(lnp, params)
    chain_list = sorted(chain_list)
    chain_list = [x for x in chain_list if np.isfinite(x[0])]
    print(chain_list[-1])
    print(chain_list[0])

    #n, lnp, params = np.array(n), np.array(lnp), np.array(params)
    #lnp = np.sort(lnp)
    #inds = np.argsort(lnp)

    #lnp = lnp[inds]
    #n = n[inds]
    #params = params[inds]

    #print(lnp[-1], max(lnp))
    #print(params[-1])

    N, t0, maxh, orbit_error = 3, 170.5, 0.01, 1.0e-20
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = utilfuncs.split_parameters(chain_list[-1][1], 3)

    x = np.linspace(-46.461309595, 1424.00096216, 65312)
    model = modeler.generate(
        N, t0, maxh, orbit_error,
        x,
        masses, radii, fluxes, u1, u2,
        a, e, inc, om, ln, ma
    )

    pylab.plot(x, model)
    pylab.savefig('./output/viewer_{0:s}.png'.format(filename.split('/')[-1].split('.')[0]))


if __name__ == '__main__':
    chain_plotter('./output/chain.dat')
    # viewnp('./output/mcmc_kep47.npz', True)
    # plotter('./output/chi2_surface_test.dat')