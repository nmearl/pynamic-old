__author__ = 'nmearl'

import numpy as np
import time
# import matplotlib
# matplotlib.use('agg')
import pylab
import photometry
import minimizer
import hammer
import utilfuncs
import argparse
import re
import multinest


def read_data(data_file):
    """Reads in photometric data. File must have at least two columns, error is optional.

    @param data_file: Path of file.
    @return: Numpy arrays of the times, data, and data error.
    """
    data_times, data_fluxes, data_yerr = [], [], []

    with open('{0}'.format(data_file), 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue

            time, flux, err = line.strip().split()[:3]
            data_times.append(float(time))
            data_fluxes.append(float(flux))
            data_yerr.append(float(err))

            # if len(data_yerr) == 0:
            # data_yerr = np.ones(len(data_fluxes))
            # x, y, data_yerr = kepler.get_fits_data('5897826', use_pdc=True, use_slc=False)

    return np.array(data_times), np.array(data_fluxes), np.array(data_yerr)


def read_input(input_file):
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

    masses = np.array([float(i.strip()) for i in input_pars[4].split()])
    radii = np.array([float(i.strip()) for i in input_pars[5].split()])
    fluxes = np.array([float(i.strip()) for i in input_pars[6].split()])
    u1 = np.array([float(i.strip()) for i in input_pars[7].split()])
    u2 = np.array([float(i.strip()) for i in input_pars[8].split()])

    a = np.array([float(i.strip()) for i in input_pars[9].split()])
    e = np.array([float(i.strip()) for i in input_pars[10].split()])
    inc = np.array([float(i.strip()) for i in input_pars[11].split()])
    om = np.array([float(i.strip()) for i in input_pars[12].split()])
    ln = np.array([float(i.strip()) for i in input_pars[13].split()])
    ma = np.array([float(i.strip()) for i in input_pars[14].split()])

    return N, t0, maxh, orbit_error, masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma


def get_random_pos(N, t0, maxh, orbit_error):
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = utilfuncs.split_parameters(utilfuncs.random_pos(N, 1)[0], N)
    return N, t0, maxh, orbit_error, masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma


def plot_model(params, x, y, yerr, rv_data):
    """Simple plot routine to visualize the modelled input parameters.

    @param params: Tuple of the parameters.
    @param x: Time data.
    @param y: Flux data.
    @param yerr: Flux error.
    """
    N, t0, maxh, orbit_error, masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = params

    mod_flux, mod_rv = photometry.multigenerate(2,
        N, t0,
        maxh, orbit_error,
        x,
        masses, radii, fluxes, u1, u2,
        a, e, inc, om, ln, ma
    )

    print("Reduced chi-square:", np.sum(((y - mod_flux) / yerr) ** 2) / (y.size - 1 - (N * 5 + (N - 1) * 6)))
    inv_sigma2 = 1.0 / (yerr ** 2 + mod_flux ** 2 * np.exp(2.0 * np.log(1.0e-10)))
    print("Custom optimized value:", -0.5 * (np.sum((y - mod_flux) ** 2 * inv_sigma2 - np.log(inv_sigma2))))

    pylab.plot(x, y, 'k+')
    pylab.plot(x, mod_flux, 'r')
    pylab.show()

    pylab.plot(x, y, 'k+')
    pylab.plot(x, mod_rv, 'r')
    pylab.show()


def main(data_file, rv_file, fit_method, input_file, nwalkers, niterations, ncores, syspars, randpars):
    """The main function mediates the reading of the data and input parameters, and starts the optimization using the
    specified method.

    @param data_file: Path to data file.
    @param fit_method: Currently: 'min' is least squares optimization, and 'mcmc' is the emcee hammer optimization. The
    'plot' option is used to simply plot the returned model based on the supplied parameters.
    @param input_file: Path to input file.
    """

    if input_file:
        params = read_input(input_file)
    elif syspars:
        print('No initial parameter set specified, randomly sampling parameter space...')
        N, t0, maxh, orbit_error = map(float, syspars)
        params = get_random_pos(int(N), t0, maxh, orbit_error)
    else:
        print('No initial parameter set specified, randomly sampling parameter space...')
        N = int(raw_input('\tEnter the number of bodies: '))
        t0 = float(raw_input('\tEnter the epoch of coordinates: '))
        maxh = float(raw_input('\tEnter the maximum time step (default: 0.01): '))
        orbit_error = float(raw_input('\tEnter the orbit error tolerance (default: 1e-20): '))

        params = get_random_pos(N, t0, maxh, orbit_error)

    if not fit_method:
        print('You have not specified a fit method, defaulting to least squares minimization.')

    if '.' in data_file:
        x, y, yerr = read_data(data_file)
    else:
        return

    rv_data = None

    if rv_file:
        rv_data = np.loadtxt(rv_file, unpack=True, usecols=(0,1,2))

    time_start = time.time()

    fname = data_file.split('/')[-1].split('.')[0]

    n = int(len(x))

    try:
        fname = re.findall(r'\d+', fname)[0]
        fname = "{0:09d}".format(int(fname))
    except:
        print('Unable to extract KID, falling back to data file name.')

    if fit_method == 'mcmc':
        hammer.generate(
            params, x[:n], y[:n], yerr[:n], rv_data,
            nwalkers, niterations, ncores, randpars, fname
        )

    elif fit_method == 'plot':
        plot_model(params, x[:n], y[:n], yerr[:n], rv_data)

    elif fit_method == 'cluster':
        params = minimizer.generate(
            params, x[:n], y[:n], yerr[:n], rv_data,
            'leastsq', ncores, fname
        )

        hammer.generate(
            params, x[:n], y[:n], yerr[:n], rv_data,
            nwalkers, niterations, ncores, randpars, fname
        )
    elif fit_method == 'multinest':
        multinest.generate(params, x, y, yerr, rv_data,
                           ncores, fname)

    else:
        minimizer.generate(
            params, x[:n], y[:n], yerr[:n], rv_data,
            fit_method, ncores, fname
        )

    print "Total time:", time.time() - time_start


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Photometric dynamical modeling code.')
    parser.add_argument('data', help='data file containing detrended light curve')
    parser.add_argument('-f', '--fit', help='fit method used to minimize',
                        choices=['multinest', 'mcmc', 'leastsq', 'nelder', 'lbfgsb', 'anneal', 'powell',
                                 'cg', 'newton', 'cobyla', 'slsqp', 'plot', 'cluster'], default='leastsq')
    parser.add_argument('-i', '--input', help='input file containing initial parameters (overrides --system)')
    parser.add_argument('-w', '--walkers', type=int, default=250,
                        help='number of walkers if using mcmc fit method ')
    parser.add_argument('-t', '--iterations', type=int, default=500,
                        help='number of iterations to perform if using mcmc fit method')
    parser.add_argument('-c', '--cores', type=int, default=1,
                        help='number of cores to utilize')
    parser.add_argument('-s', '--system', nargs=4,
                        help='four initial system parameters: N, t0, maxh, orbit_error')
    parser.add_argument('-rv', '--radial_velocity',
                        help='path to the radial velocity data')

    args = parser.parse_args()
    data_file = args.data
    rv_file = args.radial_velocity
    fit_method = args.fit
    input_file = args.input
    nwalkers = args.walkers
    niterations = args.iterations
    ncores = args.cores
    syspars = args.system
    randpars = True if not input_file else False

    main(data_file, rv_file, fit_method, input_file, nwalkers, niterations, ncores, syspars, randpars)
