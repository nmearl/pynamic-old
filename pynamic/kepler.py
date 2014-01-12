__author__ = 'nmearl'

import os
from urllib import urlretrieve, urlopen
import numpy as np
from astropy.io import fits as pyfits
import re


def get_fits_data(kep_id, qrange=None, use_pdc=False, use_slc=False):
    """
    Opens cached fits files and retrieves the time and flux data.

    :param qrange: Select desired quarter range. If None, retrieves entire available range.
    :param use_pdc: If True, uses PDC data instead of raw data.
    :return:
    """
    dirname = "./data/{0:09d}/".format(int(kep_id))

    if not os.path.exists(dirname):
        print "Retrieving data."
        _retrieve_data(kep_id, qrange)
        print "Successfully retrieved data."

    if use_slc:
        listdir = [x for x in os.listdir(dirname) if "slc" in x]
    else:
        listdir = [x for x in os.listdir(dirname) if "llc" in x]

    if qrange:
        # logger("Reading quarters {0} through {1} ({2} possible).".format(qrange[0], qrange[1], len(listdir)), "info")
        listdir = sorted(listdir)
        listdir = listdir[qrange[0] - 1:qrange[1]]
        # else:
        # logger("Reading all quarters.", "info")

    time, flags, raw_flux, raw_ferr, pdc_flux, pdc_ferr \
        = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    for filename in listdir:
        fits = pyfits.open(dirname + filename)
        time = np.append(time, fits[1].data["TIME"])
        flags = np.append(flags, fits[1].data["SAP_QUALITY"])

        med = np.median(fits[1].data["PDCSAP_FLUX"])
        pdc_flux = np.append(pdc_flux, fits[1].data["PDCSAP_FLUX"] / med)
        pdc_ferr = np.append(pdc_ferr, fits[1].data["PDCSAP_FLUX_ERR"] / np.median(fits[1].data["PDCSAP_FLUX_ERR"]))

        med = np.median(fits[1].data["SAP_FLUX"])
        raw_flux = np.append(raw_flux, fits[1].data["SAP_FLUX"] / med)
        raw_ferr = np.append(raw_ferr, fits[1].data["SAP_FLUX_ERR"] / np.median(fits[1].data["SAP_FLUX_ERR"]))

    # Remove bad data and adjust time
    raw_time = time[~np.isnan(raw_flux)]
    raw_flags = flags[~np.isnan(raw_flux)]
    #raw_time += (2454833.0 - 2455000.0)

    pdc_time = time[~np.isnan(pdc_flux)]
    pdc_flags = flags[~np.isnan(pdc_flux)]
    #pdc_time += (2454833.0 - 2455000.0)

    raw_flux = raw_flux[~np.isnan(raw_flux)]
    raw_ferr = raw_ferr[~np.isnan(raw_ferr)]

    pdc_flux = pdc_flux[~np.isnan(pdc_flux)]
    pdc_ferr = pdc_ferr[~np.isnan(pdc_ferr)]

    # Redundancy to make sure the arrays are sorted correctly
    raw_time = np.sort(raw_time)
    raw_inds = np.argsort(raw_time)

    pdc_time = np.sort(pdc_time)
    pdc_inds = np.argsort(pdc_time)

    raw_flux = raw_flux[raw_inds]
    raw_ferr = raw_ferr[raw_inds]

    pdc_flux = pdc_flux[pdc_inds]
    pdc_ferr = pdc_ferr[pdc_inds]

    # Remove median from data
    # med = np.median(raw_flux)
    # raw_flux /= med
    # raw_ferr /= med
    #
    # med = np.median(pdc_flux)
    # pdc_flux /= med
    # pdc_ferr /= med

    if use_pdc:
        return pdc_time, pdc_flux, pdc_ferr
    else:
        return raw_time, raw_flux, raw_ferr


def _retrieve_data(kep_id, qrange=None):
    kep_id = "{0:09d}".format(int(kep_id))
    dirname = "./data/{0}/".format(kep_id)

    if not os.path.exists("./data/"):
        os.mkdir("./data/")

    if not os.path.exists("./output/"):
        os.mkdir("./output/")

    if not os.path.exists(dirname):
        # logger("You do not have the data locally. Downloading Kepler data for object {0}. " \
        #        "\nPlease wait, this only needs to be done once.".format(kep_id), "info")

        os.mkdir(dirname)

        url = "http://archive.stsci.edu/pub/kepler/lightcurves//{0}/{1}/".format(kep_id[:4], kep_id[:9])
        page = urlopen(url).readlines()

        for line in page:
            if ".fits" and "lc" in line:
                fname = re.split('<a href="|.fits">', line)[1] + ".fits"
                urlretrieve(url + fname, dirname + fname)