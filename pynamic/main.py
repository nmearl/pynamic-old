__author__ = 'nmearl'

# import matplotlib
# matplotlib.use('agg')
import numpy as np
import pylab
import minimizer
import hammer
import utilfuncs
import argparse
try:
    import multinest
except:
    pass


def read_input(input_file):
    temp_dict = {}

    with open('{0}'.format(input_file), 'r') as f:
        for line in f:
            if line[0] is '#':
                continue

            nline = line.strip().split('#')[0]
            n, v = nline.strip().split('=')
            temp_dict[n.strip()] = v.strip()

            # input_pars = [line.strip() for line in f.readlines() if line[0] is not '#']

    photo_data_file = temp_dict['photo_data_file']
    rv_data_file = temp_dict['rv_data_file']
    rv_body = int(temp_dict['rv_body']) if temp_dict['rv_body'] and temp_dict['rv_body'] > 0 else 1
    rv_corr = None  # float(temp_dict['rv_corr'])
    nwalkers = int(temp_dict['nwalkers'])
    out_prefix = temp_dict['out_prefix']

    nbodies = int(temp_dict['nbodies'])
    epoch = float(temp_dict['epoch'])
    max_h = float(temp_dict['max_h'])
    orbit_error = float(temp_dict['orbit_error'])

    masses = np.array([float(x) for x in temp_dict['masses'].split()])
    radii = np.array([float(x) for x in temp_dict['radii'].split()])
    fluxes = np.array([float(x) for x in temp_dict['fluxes'].split()])
    u1 = np.array([float(x) for x in temp_dict['u1'].split()])
    u2 = np.array([float(x) for x in temp_dict['u2'].split()])

    a = np.array([float(x) for x in temp_dict['a'].split()])
    e = np.array([float(x) for x in temp_dict['e'].split()])
    inc = np.array([float(x) for x in temp_dict['inc'].split()])
    om = np.array([float(x) for x in temp_dict['om'].split()])
    ln = np.array([float(x) for x in temp_dict['ln'].split()])
    ma = np.array([float(x) for x in temp_dict['ma'].split()])

    sys_pars = [photo_data_file, rv_data_file, nwalkers, out_prefix]
    mod_pars = [nbodies, epoch, max_h, orbit_error, rv_body, rv_corr]
    body_pars = [masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma]

    return sys_pars, mod_pars, body_pars


def plot_model(mod_pars, body_pars, photo_data, rv_data):
    print("Here")
    mod_flux, mod_rv = utilfuncs.model(mod_pars, body_pars, photo_data[0], photo_data[0])
    print("here2")
    # print("Reduced chi-square:",
    #       np.sum(((photo_data[1] - mod_flux) / photo_data[2]) ** 2) /
    #       (photo_data[2].size - 1 - (mod_pars[0] * 5 + (mod_pars[0] - 1) * 6)))
    #
    # inv_sigma2 = 1.0 / (photo_data[2] ** 2 + mod_flux ** 2 * np.exp(2.0 * np.log(1.0e-10)))
    # print("Custom optimized value:",
    #       -0.5 * (np.sum((photo_data[1] - mod_flux) ** 2 * inv_sigma2 - np.log(inv_sigma2))))

    pylab.plot(photo_data[0], photo_data[1], 'k+')
    pylab.plot(photo_data[0], mod_flux, 'r')
    pylab.show()

    pylab.plot(rv_data[0], rv_data[1], 'o')
    pylab.plot(photo_data[0], mod_rv, 'r')
    pylab.show()


def main(input_file, fit_method, nprocs):
    sys_pars, mod_pars, body_pars = read_input(input_file)
    photo_data_file, rv_data_file, nwalkers, out_prefix = sys_pars

    photo_data = np.loadtxt(photo_data_file, unpack=True, usecols=(0, 1, 2))

    rv_data = np.loadtxt(rv_data_file, unpack=True, usecols=(0, 1, 2)) \
        if rv_data_file else np.zeros((3, 0))

    if fit_method == 'mcmc':
        hammer.generate(mod_pars, body_pars, photo_data, rv_data, nwalkers, nprocs, out_prefix)
    elif fit_method == 'multinest':
        multinest.generate(mod_pars, body_pars, photo_data, rv_data, nprocs, out_prefix)
    elif fit_method == 'cluster':
        fparams = minimizer.generate(mod_pars, body_pars, photo_data, rv_data, 'leastsq', nprocs, out_prefix)
        hammer.generate(mod_pars, fparams, photo_data, rv_data, nwalkers, nprocs, out_prefix)
    elif fit_method == 'plot':
        plot_model(mod_pars, body_pars, photo_data, rv_data)
    else:
        minimizer.generate(mod_pars, body_pars, photo_data, rv_data, fit_method, nprocs, out_prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Photometric dynamical modeling code.')
    parser.add_argument('input_file', help='input file containing all necessary parameters')
    # parser.add_argument('data', help='data file containing detrended light curve')
    # parser.add_argument('rv', help='path to the radial velocity data')
    parser.add_argument('-f', '--fit', help='fit method used to minimize',
                        choices=['multinest', 'mcmc', 'leastsq', 'nelder', 'lbfgsb', 'anneal', 'powell',
                                 'cg', 'newton', 'cobyla', 'slsqp', 'plot', 'cluster'], default='leastsq')
    # parser.add_argument('-i', '--input', help='input file containing initial parameters (overrides --system)')
    # parser.add_argument('-w', '--walkers', type=int, default=250,
    #                     help='number of walkers if using mcmc fit method ')
    # parser.add_argument('-t', '--iterations', type=int, default=500,
    #                     help='number of iterations to perform if using mcmc fit method')
    parser.add_argument('-p', '--procs', type=int, default=1,
                        help='number of processors to utilize')
    # parser.add_argument('-s', '--system', nargs=4,
    #                     help='four initial system parameters: N, t0, maxh, orbit_error')
    # parser.add_argument('-rvb', '--rv_body', type=int, default=1,
    #                     help='which body is the radial velocity data for')

    args = parser.parse_args()
    input_file = args.input_file
    # data_file = args.data
    # rv_file = args.radial_velocity
    fit_method = args.fit
    # input_file = args.input
    # nwalkers = args.walkers
    # niterations = args.iterations
    nprocs = args.procs
    # syspars = args.system
    # randpars = True if not input_file else False
    # rv_body = args.rv_body

    # cProfile.run('main(data_file, rv_file, fit_method, input_file, nwalkers, niterations, ncores, syspars, randpars)')
    # main(data_file, rv_file, rv_body, fit_method, input_file, nwalkers, niterations, ncores, syspars, randpars)
    main(input_file, fit_method, nprocs)