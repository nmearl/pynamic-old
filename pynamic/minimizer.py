__author__ = 'nmearl'

import numpy as np
from lmfit import minimize, report_fit
from lmfit import Parameters as lmParameters
import photometry
import utilfuncs

twopi = 2.0 * np.pi


def per_iteration(lmparams, i, resids, mod_pars, photo_data, rv_data, *args, **kws):
    if i%10 == 0.0:
        ncores, fname = args

        params = utilfuncs.get_lmfit_parameters(mod_pars, lmparams)

        redchisqr = utilfuncs.reduced_chisqr(mod_pars, params, photo_data, rv_data)
        utilfuncs.iterprint(mod_pars, params, 0.0, redchisqr, 0.0, 0.0)
        utilfuncs.report_as_input(mod_pars, params, fname)


def residual(params, mod_pars, photo_data, rv_data, ncores, *args):
    params = utilfuncs.get_lmfit_parameters(mod_pars, params)

    mod_flux, mod_rv = utilfuncs.model(mod_pars, params, photo_data[0], rv_data[0], ncores)
    weighted = ((mod_flux - photo_data[1]) / photo_data[2])

    return np.append(weighted, ((mod_rv - rv_data[1]) / rv_data[2]))


def generate(mod_pars, body_pars, photo_data, rv_data, fit_method, ncores, fname):
    nbodies, epoch, max_h, orbit_error, rv_body, rv_corr = mod_pars
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = body_pars

    lmparams = lmParameters()
    # lmparams.add('N', value=N, vary=False)
    # lmparams.add('t0', value=t0, vary=False)
    # lmparams.add('maxh', value=maxh, vary=False)
    # lmparams.add('orbit_error', value=orbit_error, vary=False)

    for i in range(nbodies):
        lmparams.add('mass_{0}'.format(i), value=masses[i], min=0.0, max=0.1)
        lmparams.add('radius_{0}'.format(i), value=radii[i], min=0.0, max=1.0)
        lmparams.add('flux_{0}'.format(i), value=fluxes[i], min=0.0, max=1.0)
        lmparams.add('u1_{0}'.format(i), value=u1[i], min=0.0, max=1.0)
        lmparams.add('u2_{0}'.format(i), value=u2[i], min=0.0, max=1.0)

        # if i < N-1:
        #     params['flux_{0}'.format(i)].vary = False
        #     params['u1_{0}'.format(i)].vary = False
        #     params['u2_{0}'.format(i)].vary = False

        if i > 0:
            lmparams.add('a_{0}'.format(i), value=a[i - 1], min=0.0, max=10.0)
            lmparams.add('e_{0}'.format(i), value=e[i - 1], min=0.0, max=1.0)
            lmparams.add('inc_{0}'.format(i), value=inc[i - 1], min=0.0, max=np.pi)
            lmparams.add('om_{0}'.format(i), value=om[i - 1], min=0.0, max=twopi)
            lmparams.add('ln_{0}'.format(i), value=ln[i - 1], min=0.0, max=twopi)
            lmparams.add('ma_{0}'.format(i), value=ma[i - 1], min=0.0, max=twopi)

    print('Generating maximum likelihood values...')
    results = minimize(residual, lmparams, args=(mod_pars, photo_data, rv_data, ncores, fname),
                       iter_cb=per_iteration, method=fit_method)

    # Save the final outputs
    print "Writing report..."
    report_fit(results.params)
    utilfuncs.report_as_input(mod_pars, utilfuncs.get_lmfit_parameters(mod_pars, results.params), fname)

    # Return best fit values
    return utilfuncs.get_lmfit_parameters(mod_pars, results.params)