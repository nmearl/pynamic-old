__author__ = 'nmearl'

import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import sys

if sys.platform == 'darwin':
    # print("It seems you're on mac, loading mac libraries...")
    lib = ctypes.cdll.LoadLibrary('./lib/photodynam-mac.so')
else:
    # print("It seems you're on linux, loading linux libraries...")
    lib = ctypes.cdll.LoadLibrary('./lib/photodynam.so')

start = lib.start

start.argtypes = [
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


def generate(N, t0, maxh, orbit_error, in_times, mass, radii, flux, u1, u2, a, e, inc, om, ln, ma):
    #in_times = np.linspace(in_times[0], in_times[-1], len(in_times))
    fluxes = np.zeros(len(in_times))

    print('\tPassing to Carter code...')

    start(
        fluxes,
        N, t0, maxh, orbit_error,
        len(in_times), np.array(in_times),
        np.array(mass), np.array(radii), np.array(flux), np.array(u1), np.array(u2),
        np.array(a), np.array(e), np.array(inc), np.array(om), np.array(ln), np.array(ma)
    )

    print('\tReturning...')

    return fluxes
