# filename: codebase/camb_cmb_spectrum.py
import os
import numpy as np
import pandas as pd

# Ensure data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

import camb
from camb import model, initialpower

def compute_cmb_tt_spectrum():
    r"""
    Computes the CMB temperature power spectrum for a non-flat Lambda CDM cosmology using CAMB.

    Returns:
        l (ndarray): Multipole moments [unitless]
        tt (ndarray): Temperature power spectrum l(l+1)C_l^{TT}/(2pi) [unit: μK^2]
    """
    # Cosmological parameters
    H0 = 67.3  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [unitless]
    omch2 = 0.122  # Cold dark matter density [unitless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.05  # Curvature [unitless]
    tau = 0.06  # Optical depth [unitless]
    As = 2e-9  # Scalar amplitude [unitless]
    ns = 0.965  # Scalar spectral index [unitless]

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(3000, lens_potential_accuracy=0)
    pars.WantTensors = False

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000)
    totCL = powers['total']

    # l values: CAMB returns from l=0,1,...,lmax
    l = np.arange(totCL.shape[0])
    # Extract TT spectrum: l(l+1)C_l^{TT}/(2pi) in μK^2
    tt = totCL[:, 0]  # TT spectrum

    # Restrict to l=2 to l=3000
    lmin = 2
    lmax = 3000
    l = l[lmin:lmax+1]
    tt = tt[lmin:lmax+1]

    return l, tt

def save_spectrum_to_csv(l, tt, filename):
    r"""
    Saves the multipole moments and TT spectrum to a CSV file.

    Args:
        l (ndarray): Multipole moments [unitless]
        tt (ndarray): Temperature power spectrum [μK^2]
        filename (str): Output CSV filename
    """
    df = pd.DataFrame({'l': l, 'TT': tt})
    df.to_csv(filename, index=False)
    print("Saved CMB TT power spectrum to " + filename)
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())
    print("Total number of multipoles: " + str(len(df)))

if __name__ == "__main__":
    l, tt = compute_cmb_tt_spectrum()
    save_spectrum_to_csv(l, tt, "data/result.csv")
