__author__ = 'nmearl'

import numpy as np
from lmfit import minimize, Parameters, report_fit
import modeler
import utilfuncs

twopi = 2.0 * np.pi


def per_iteration(params, i, resids, x, y, yerr, *args, **kws):
    pass


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


def generate(in_params, x, y, yerr, fit_method, fname):
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
            params.add('ln_{0}'.format(i), value=ln[i - 1], min=0.0, max=np.pi)
            params.add('ma_{0}'.format(i), value=ma[i - 1], min=0.0, max=twopi)

    # params['flux_2'].min= 0.8
    # params['flux_2'].max = 1.0
    # params['flux_2'].vary = False
    # params['flux_2'].expr = '1.0 - flux_0 - flux_1'
    # params['ln_2'].vary = False

    print('Generating maximum likelihood values...')
    results = minimize(residual, params, args=(x, y, yerr), method=fit_method)

    # for i in range(1):
    #     print "Run {0}".format(i + 1)
    #result = minimize(residual, result.params, args=(x, y, yerr))#, method='anneal')

    # Give some info on the fit
    # print "Reduced chi-squared:", result.redchi
    # print "Chi-squared:", result.chisqr
    report_fit(params)

    # Save the final outputs
    print "Writing report..."
    utilfuncs.report_as_input(N, t0, maxh, orbit_error, _get_parameters(results.params), fname)
