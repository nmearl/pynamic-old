__author__ = 'nmearl'

import numpy as np
from lmfit import minimize, Parameters, report_fit
import photometry
import utilfuncs

twopi = 2.0 * np.pi


def per_iteration(params, i, resids, x, y, yerr, rv_data, *args, **kws):
    if i%10 == 0.0:
        ncores, fname = args

        params = utilfuncs.get_lmfit_parameters(params)

        redchisqr = utilfuncs.reduced_chisqr(params, x, y, yerr)
        utilfuncs.iterprint(params, 0.0, redchisqr, 0.0, 0.0)
        utilfuncs.report_as_input(params, fname)


def residual(params, x, y, yerr, rv_data, ncores, *args):
    params = utilfuncs.get_lmfit_parameters(params)
    mod_flux, _ = utilfuncs.model(params, x)
    _, mod_rv = utilfuncs.model(params, rv_data[0])

    weighted = (mod_flux - y) / yerr

    if rv_data is not None:
        return weighted + (mod_rv - rv_data[1]) / rv_data[2]

    return weighted


def generate(params, x, y, yerr, rv_data, fit_method, ncores, fname):
    N, t0, maxh, orbit_error, masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = params

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
            params.add('om_{0}'.format(i), value=om[i - 1], min=0.0, max=twopi)
            params.add('ln_{0}'.format(i), value=ln[i - 1], min=0.0, max=twopi)
            params.add('ma_{0}'.format(i), value=ma[i - 1], min=0.0, max=twopi)

    print('Generating maximum likelihood values...')
    results = minimize(residual, params, args=(x, y, yerr, rv_data, ncores, fname),
                       iter_cb=per_iteration, method=fit_method)

    # Save the final outputs
    print "Writing report..."
    report_fit(results.params)
    utilfuncs.report_as_input(utilfuncs.get_lmfit_parameters(params), fname)

    # Return best fit values
    return utilfuncs.get_lmfit_parameters(results.params)
