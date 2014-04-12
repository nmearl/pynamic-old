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
    ndpointer(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
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
    sub_fluxes, sub_rv, N, t0, maxh, orbit_error, in_times_size, in_times, \
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma = inputs

    start(
        sub_fluxes, sub_rv,
        N, t0, maxh, orbit_error,
        in_times_size, in_times,
        masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma
    )

    return sub_fluxes, sub_rv


def multigenerate(ncores, N, t0, maxh, orbit_error, in_times, masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma):
    chunk = int(len(in_times) / ncores)

    inputs = [
        [
            np.zeros(len(in_times[i:i + chunk])),
            np.zeros(len(in_times[i:i + chunk])),
            N, t0, maxh, orbit_error,
            len(in_times[i:i + chunk]), np.array(in_times[i:i + chunk]),
            masses, radii, fluxes, u1, u2,
            a, e, inc, om, ln, ma
        ]
        for i in range(0, len(in_times), chunk)
    ]

    p = Pool(ncores)

    result = p.map(run, inputs)

    p.close()
    p.join()

    # fluxes = np.array([])
    # for sub_fluxes in result:
    #     fluxes = np.append(fluxes, sub_fluxes)

    result = np.array(result)

    # tot_rv = np.concatenate(result[:, 1]) * 1731 - 27.278 - 0.26
    # pylab.plot(in_times, tot_rv)
    #
    # time, data, err = np.loadtxt('data/005897_rv.dat', unpack=True)
    #
    # pylab.plot(time[np.argsort(time)], data[np.argsort(time)], 'o')
    # pylab.show()

    return np.concatenate(result[:, 0]), np.concatenate(result[:, 1]) * 1731 - 27.278 - 0.26


def generate(N, t0, maxh, orbit_error, in_times, masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma):

    tot_flux = np.zeros(len(in_times))
    tot_rv = np.zeros(len(in_times))

    start(
        tot_flux, tot_rv,
        N, t0, maxh, orbit_error,
        len(in_times), in_times,
        masses, radii, fluxes, u1, u2,
        a, e, inc, om, ln, ma
    )

    return tot_flux, tot_rv * 1731 - 27.278 - 0.26
