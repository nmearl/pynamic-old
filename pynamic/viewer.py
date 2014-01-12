__author__ = 'nmearl'

import numpy as np
import pylab
from matplotlib import gridspec


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


if __name__ == '__main__':
    viewnp('./output/mcmc_kep47.npz', True)
    # plotter('./output/chi2_surface_test.dat')