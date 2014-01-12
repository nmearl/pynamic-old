__author__ = 'nmearl'

import sys
import numpy as np
import time
import pylab
import modeler
import minimizer
import hammer
import kepler


def _read_data(data_file):
    """Reads in photometric data. File must have at least two columns, error is optional.

    @param data_file: Path of file.
    @return: Numpy arrays of the times, data, and data error.
    """
    data_times, data_fluxes, data_yerr = [], [], []

    with open('{0}'.format(data_file), 'r') as f:
        for line in f.readlines():
            time, flux, err = line.strip().split()[:3]
            data_times.append(float(time))
            data_fluxes.append(float(flux))
            data_yerr.append(float(err))

            # if len(data_yerr) == 0:
            # data_yerr = np.ones(len(data_fluxes))
            # x, y, data_yerr = kepler.get_fits_data('5897826', use_pdc=True, use_slc=False)

    return np.array(data_times), np.array(data_fluxes), np.array(data_yerr)


def _read_input(input_file):
    """Reads in the formatted input file.

    @param input_file: Path of file.
    @return: Tuple containing all 15 parameter data.
    """
    with open('{0}'.format(input_file), 'r') as f:
        input_pars = [line.strip() for line in f.readlines() if line[0] is not '#']

    N = [int(i.strip()) for i in input_pars[0].split('#')][0]
    t0 = [float(i.strip()) for i in input_pars[1].split('#')][0]
    maxh = [float(i.strip()) for i in input_pars[2].split('#')][0]
    orbit_error = [float(i.strip()) for i in input_pars[3].split('#')][0]

    masses = [float(i.strip()) for i in input_pars[4].split()]
    radii = [float(i.strip()) for i in input_pars[5].split()]
    fluxes = [float(i.strip()) for i in input_pars[6].split()]
    u1 = [float(i.strip()) for i in input_pars[7].split()]
    u2 = [float(i.strip()) for i in input_pars[8].split()]

    a = [float(i.strip()) for i in input_pars[9].split()]
    e = [float(i.strip()) for i in input_pars[10].split()]
    inc = [float(i.strip()) for i in input_pars[11].split()]
    om = [float(i.strip()) for i in input_pars[12].split()]
    ln = [float(i.strip()) for i in input_pars[13].split()]
    ma = [float(i.strip()) for i in input_pars[14].split()]

    return N, t0, maxh, orbit_error, masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma


def plot_model(params, x, y, yerr):
    """Simple plot routine to visualize the modelled input parameters.

    @param params: Tuple of the parameters.
    @param x: Time data.
    @param y: Flux data.
    @param yerr: Flux error.
    """
    N, t0, maxh, orbit_error, masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = params

    y_mod = modeler.generate(
        N, t0,
        maxh, orbit_error,
        x,
        masses, radii, fluxes, u1, u2,
        a, e, inc, om, ln, ma
    )

    pylab.plot(x, y, 'k+')
    pylab.plot(x, y_mod, 'r')

    pylab.show()


def main(input_file, data_file, fit_method):
    """The main function mediates the reading of the data and input parameters, and starts the optimization using the
    specified method.

    @param input_file: Path to input file.
    @param data_file: Path to data file.
    @param fit_method: Currently: 'min' is least squares optimization, and 'mcmc' is the emcee hammer optimization. The
    'plot' option is used to simply plot the returned model based on the supplied parameters.
    """
    params = _read_input(input_file)

    if '.' in data_file:
        x, y, yerr = _read_data(data_file)
        # else:
    #     x, y, yerr = kepler.get_fits_data('5897826', use_pdc=True)

    n = int(len(x))

    print "Beginning optimization."

    time_start = time.time()

    if fit_method == 'min':
        minimizer.generate(
            params, x[:n], y[:n], yerr[:n]
        )

    elif fit_method == 'mcmc':
        hammer.generate(
            params, x[:n], y[:n], yerr[:n],
        )

    elif fit_method == 'plot':
        plot_model(params, x[:n], y[:n], yerr[:n])

    print "Total time:", time.time() - time_start


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args[0], args[1], args[2])
    # main('./input.dat', './data/kid010020423.Q99', 'ls')