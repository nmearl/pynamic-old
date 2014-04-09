__author__ = 'nmearl'

import numpy as np
from lmfit import minimize, Parameters, report_fit
import photometry
import utilfuncs

twopi = 2.0 * np.pi


def per_iteration(params, i, resids, x, y, yerr, *args, **kws):
    if i%100 == 0.0:
        ncores, fname = args
        N, t0, maxh, orbit_error = params['N'].value, params['t0'].value, \
                                   params['maxh'].value, params['orbit_error'].value
        split_params = _get_parameters(params)
        redchisqr = utilfuncs.reduced_chisqr(np.concatenate(split_params), x, y, yerr, N, t0, maxh, orbit_error)
        utilfuncs.iterprint(N, np.concatenate(_get_parameters(params)), 0.0, redchisqr, 0.0, 0.0)
        utilfuncs.report_as_input(N, t0, maxh, orbit_error, _get_parameters(params), fname)


def _get_parameters(params, full=False):
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

    if not full:
        return masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma
    else:
        return params['N'].value, params['t0'].value, params['maxh'].value, params['orbit_error'].value, \
               masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma


def fitfunc(params, x, ncores):
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = _get_parameters(params)

    out_fluxes = photometry.multigenerate(ncores,
        params['N'].value, params['t0'].value,
        params['maxh'].value, params['orbit_error'].value,
        x,
        masses, radii, fluxes, u1, u2,
        a, e, inc, om, ln, ma
    )

    return out_fluxes


def residual(params, x, y, yerr, ncores, *args):
    model = fitfunc(params, x, ncores)
    weighted = (model - y) / yerr
    # per_iteration(params, y, yerr, model, chi2)
    return weighted


def generate(in_params, x, y, yerr, fit_method, ncores, fname):
    N, t0, maxh, orbit_error, masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = in_params

    params = Parameters()
    params.add('N', value=N, vary=False)
    params.add('t0', value=t0, vary=False)
    params.add('maxh', value=maxh, vary=False)
    params.add('orbit_error', value=orbit_error, vary=False)

    for i in range(N):
        params.add('mass_{0}'.format(i), value=masses[i], min=0.0, max=0.1)
        params.add('radius_{0}'.format(i), value=radii[i], min=0.0, max=1.0)
        params.add('flux_{0}'.format(i), value=fluxes[i], min=0.0, max=1.0)
        params.add('u1_{0}'.format(i), value=u1[i], min=0.0, max=1.0)
        params.add('u2_{0}'.format(i), value=u2[i], min=0.0, max=1.0)

        # if i < N-1:
        #     params['flux_{0}'.format(i)].vary = False
        #     params['u1_{0}'.format(i)].vary = False
        #     params['u2_{0}'.format(i)].vary = False

        if i > 0:
            params.add('a_{0}'.format(i), value=a[i - 1], min=0.0, max=10.0)
            params.add('e_{0}'.format(i), value=e[i - 1], min=0.0, max=1.0)
            params.add('inc_{0}'.format(i), value=inc[i - 1], min=0.0, max=np.pi)
            params.add('om_{0}'.format(i), value=om[i - 1], min=-twopi, max=twopi)
            params.add('ln_{0}'.format(i), value=ln[i - 1], min=-twopi, max=twopi)
            params.add('ma_{0}'.format(i), value=ma[i - 1], min=0.0, max=twopi)

    print('Generating maximum likelihood values...')
    results = minimize(residual, params, args=(x, y, yerr, ncores, fname), iter_cb=per_iteration, method=fit_method)

    # Save the final outputs
    print "Writing report..."
    # report_fit(params)
    utilfuncs.report_as_input(N, t0, maxh, orbit_error, _get_parameters(results.params), fname)

    # Return best fit values
    return _get_parameters(results.params, full=True)
