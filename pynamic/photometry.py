__author__ = 'nmearl'

import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import sys
from multiprocessing import Pool
import pylab


if sys.platform == 'darwin':
    # print("It seems you're on mac, loading mac libraries...")
    lib = ctypes.cdll.LoadLibrary('lib/photodynam-mac.so')
else:
    # print("It seems you're on linux, loading linux libraries...")
    lib = ctypes.cdll.LoadLibrary('lib/photodynam.so')

start = lib.start

start.argtypes = [
    ndpointer(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double)
]


def run(inputs):
    time, time_size, \
    N, t0, maxh, orbit_error, \
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma, \
    sub_flux, sub_rv = inputs

    start(
        time, time_size,
        N, t0, maxh, orbit_error,
        masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma,
        sub_flux, sub_rv
    )

    return sub_flux, sub_rv


def generate(params, time, ncores=1):
    N, t0, maxh, orbit_error, \
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = params

    time_chunks = np.array_split(time, ncores)

    inputs = [
        [
            time_chunks[i], len(time_chunks[i]),
            N, t0, maxh, orbit_error,
            masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma,
            np.zeros(len(time_chunks[i])),
            np.zeros(len(time_chunks[i]))
        ]
        for i in range(ncores)
    ]

    if ncores > 1:
        p = Pool(ncores)

        result = p.map(run, inputs)

        p.close()
        p.join()
    else:
        result = map(run, inputs)

    # fluxes = np.array([])
    # for sub_fluxes in result:
    #     fluxes = np.append(fluxes, sub_fluxes)

    result = np.array(result)

    return np.concatenate(result[:, 0]), np.concatenate(result[:, 1]) * 1731 - 27.278 - 0.26


# def generate(time, params):
#     N, t0, maxh, orbit_error, \
#     masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = params
#
#     tot_flux = np.zeros(len(time))
#     tot_rv = np.zeros(len(time))
#
#     start(
#         tot_flux, tot_rv,
#         N, t0, maxh, orbit_error,
#         len(in_times), in_times,
#         masses, radii, fluxes, u1, u2,
#         a, e, inc, om, ln, ma
#     )
#
#     return tot_flux, tot_rv * 1731 - 27.278 - 0.26
