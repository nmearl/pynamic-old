__author__ = 'nmearl'

import numpy as np
from lmfit import minimize, Parameters, report_fit, fit_report
import pylab
from matplotlib import gridspec
import modeler
# import surface
# from itertools import product
import os
from multiprocessing import Pool
import surface as surf

twopi = 2.0 * np.pi


def per_iteration(params, i, resids, x, y, yerr, *args, **kws):
    chi2 = np.sum((resids) ** 2)
    write_out(params, chi2)


def _get_parameters(params):
    masses = [params['mass_{0}'.format(i)].value for i in range(params['N'].value)]
    radii = [params['radius_{0}'.format(i)].value for i in range(params['N'].value)]
    fluxes = [params['flux_{0}'.format(i)].value for i in range(params['N'].value)]
    u1 = [params['u1_{0}'.format(i)].value for i in range(params['N'].value)]
    u2 = [params['u2_{0}'.format(i)].value for i in range(params['N'].value)]

    a = [params['a_{0}'.format(i)].value for i in range(1, params['N'].value)]
    e = [params['e_{0}'.format(i)].value for i in range(1, params['N'].value)]
    inc = [params['inc_{0}'.format(i)].value for i in range(1, params['N'].value)]
    om = [params['om_{0}'.format(i)].value for i in range(1, params['N'].value)]
    ln = [params['ln_{0}'.format(i)].value for i in range(1, params['N'].value)]
    ma = [params['ma_{0}'.format(i)].value for i in range(1, params['N'].value)]

    return masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma


def fitfunc(params, x):
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = _get_parameters(params)

    out_fluxes = modeler.generate(
        params['N'].value, params['t0'].value,
        params['maxh'].value, params['orbit_error'].value,
        x,
        masses, radii, fluxes, u1, u2,
        a, e, inc, om, ln, ma
    )

    return out_fluxes


def residual(params, x, y, yerr, *args):
    model = fitfunc(params, x)
    weighted = (model - y) / yerr
    # per_iteration(params, y, yerr, model, chi2)
    return weighted


def generate(in_params, x, y, yerr):
    N, t0, maxh, orbit_error, masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = in_params

    params = Parameters()
    params.add('N', value=N, vary=False)
    params.add('t0', value=t0, vary=False)
    params.add('maxh', value=maxh, vary=False)
    params.add('orbit_error', value=orbit_error, vary=False)

    for i in range(N):
        params.add('mass_{0}'.format(i), value=masses[i], min=0.0, max=0.01)
        params.add('radius_{0}'.format(i), value=radii[i], min=0.0, max=0.1)
        params.add('flux_{0}'.format(i), value=fluxes[i], min=0.0, max=1.0)
        params.add('u1_{0}'.format(i), value=u1[i], min=0.0, max=1.0)
        params.add('u2_{0}'.format(i), value=u2[i], min=0.0, max=1.0)

        # if i < N-1:
        #     params['flux_{0}'.format(i)].vary = False
        #     params['u1_{0}'.format(i)].vary = False
        #     params['u2_{0}'.format(i)].vary = False

        if i > 0:
            params.add('a_{0}'.format(i), value=a[i - 1], min=0.0, max=1.0)
            params.add('e_{0}'.format(i), value=e[i - 1], min=0.0, max=1.0)
            params.add('inc_{0}'.format(i), value=inc[i - 1], min=0.0, max=np.pi)
            params.add('om_{0}'.format(i), value=om[i - 1], min=0.0, max=twopi)
            params.add('ln_{0}'.format(i), value=ln[i - 1], min=-twopi, max=twopi)
            params.add('ma_{0}'.format(i), value=ma[i - 1], min=0.0, max=twopi)

    # params['flux_2'].min= 0.8
    # params['flux_2'].max = 1.0
    # params['flux_2'].vary = False
    # params['flux_2'].expr = '1.0 - flux_0 - flux_1'
    # params['ln_2'].vary = False

    print('Generating maximum likelihood values...')
    result = minimize(residual, params, args=(x, y, yerr), method='anneal')

    for i in range(1):
        print "Run {0}".format(i + 1)
        #result = minimize(residual, result.params, args=(x, y, yerr))#, method='anneal')

    # Give some info on the fit
    # print "Reduced chi-squared:", result.redchi
    # print "Chi-squared:", result.chisqr
    report_fit(params)

    # Save the final outputs
    model = fitfunc(params, x)
    print "Saving data..."
    np.savez('./output/min_kep126', x, y, yerr, model)
    print "Writing report..."
    report_out(x, y, model, result, params, fit_report(params))

    # Generate surfaces
    # print("Generating chi-squared surfaces...")
    # surface(params, 'radius_0', x, y, yerr)
    # surf.generate(params, 1000, x, y, yerr, N)

    # Plot the outputs
    plotter(x, y, model, result.residual)


def _minimize(par_dict, x, y, yerr):
    # try:
    params = Parameters()

    for key, val in par_dict.items():
        params.add(key, value=val[0], min=val[1], max=val[2], vary=val[3])

    result = minimize(residual, params, iter_cb=per_iteration, args=(x, y, yerr))#, method='lbfgsb')
    write_out(result)
    # except:
    #     print("Error occured while writing out.")


def surface(params, par_name, x, y, yerr):
    par_range = np.linspace(params[par_name] * 0.8, params[par_name] * 1.2, 100)
    params[par_name].vary = False

    pool = Pool(processes=2)

    for par_val in par_range:
        params[par_name].value = par_val

        par_dict = {}

        for name, par in params.items():
            par_dict[name] = (par.value, par.min, par.max, par.vary)

        # result = minimize(residual, params, iter_cb=per_iteration, args=(x, y, yerr), method='anneal')
        # write_out(result)

        pool.apply_async(
            _minimize, # (residual, params, args=(x, y, yerr)),
            args=(par_dict, x, y, yerr),
            # callback=write_out
        )

    pool.close()
    pool.join()


def write_out(params, chi2):
    # params = result.params

    floc = './output/chi2_surface_test.dat'
    status = 'a' if os.path.exists(floc) else 'w'

    with open(floc, status) as f:
        if status == 'w':
            for name, par in params.items():
                f.write('{0} '.format(name))

            f.write('{0}\n'.format('chi_squared'))

        for name, par in params.items():
            f.write('{0} '.format(par.value))

        f.write('{0}\n'.format(chi2))


def plotter(x, data, model, resids):
    fig = pylab.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    ax1 = pylab.subplot(gs[0])
    ax1.plot(x, data, 'k+')
    ax1.plot(x, data + resids, 'r')
    ax1.plot(x, model, 'b')

    ax2 = pylab.subplot(gs[1])
    ax2.plot(x, resids)
    pylab.savefig('./output/lc_kep126_min.eps')

    pylab.xlabel('Flux')
    pylab.xlabel('Time (BJD)')
    pylab.savefig('./output/out_fig.eps')


def report_out(x, data, model, result, params, report):
    GMsun = 2.959122083E-4 # AU**3/day**2
    Rsun = 0.00465116 # AU

    with open('./output/report_kep126_min.out', 'w') as f:
        f.write("Reduced chi-squared: {0}\n".format(result.redchi))

        for key, para in params.items():
            if not para.vary:
                f.write(
                    "{0:12s}\t{1:20.11e}\tfixed\n".format(
                        para.name, para.value
                    )
                )
            elif 'mass' in para.name:
                f.write(
                    "{0:12s}\t{1:20.11e}\t{2:20.11e}\n".format(
                        para.name, para.value / GMsun, (para.stderr / para.value) * (para.value / GMsun)
                    )
                )
            elif 'radius' in para.name:
                f.write(
                    "{0:12s}\t{1:20.11e}\t{2:20.11e}\n".format(
                        para.name, para.value / Rsun, (para.stderr / para.value) * (para.value / Rsun)
                    )
                )
            elif any(ang in para.name for ang in ['inc', 'om', 'ln', 'ma']):
                f.write(
                    "{0:12s}\t{1:20.11e}\t{2:20.11e}\n".format(
                        para.name, np.rad2deg(para.value), (para.stderr / para.value) * np.rad2deg(para.value)
                    )
                )
            else:
                f.write(
                    "{0:12s}\t{1:20.11e}\t{2:20.11e}\n".format(
                        para.name, para.value, para.stderr
                    )
                )

        for line in report:
            f.write(line)

    with open('./output/data.out', 'w') as f:
        for i in range(len(x)):
            f.write(
                '{0:15e} {1:15e} {2:15e} {3:15e}\n'.format(
                    x[i], model[i], data[i], result.residual[i]
                )
            )
