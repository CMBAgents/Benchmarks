# filename: codebase/cmb_spectrum.py
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
    Compute the CMB temperature power spectrum for the specified flat Lambda CDM cosmology.

    Returns:
        l (np.ndarray): Multipole moments [unitless]
        TT (np.ndarray): Temperature power spectrum l(l+1)C_l^{TT}/(2pi) [unit: μK^2]
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.02  # Omega_b h^2 [unitless]
    omch2 = 0.122  # Omega_c h^2 [unitless]
    mnu = 0.06  # Sum of neutrino masses [eV]
    omk = 0.0  # Curvature [unitless]
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
    totCL = powers['total']  # Shape: (lmax+1, 4) for TT, EE, BB, TE

    # l values: CAMB returns from l=0, so index 2 corresponds to l=2
    l = np.arange(2, 3001)  # l=2 to l=3000 [unitless]
    # totCL[:,0] is TT, already in μK^2 units
    TT = totCL[2:3001, 0]  # [μK^2]

    return l, TT

def save_spectrum_to_csv(l, TT, filename):
    r"""
    Save the multipole moments and temperature power spectrum to a CSV file.

    Args:
        l (np.ndarray): Multipole moments [unitless]
        TT (np.ndarray): Temperature power spectrum [μK^2]
        filename (str): Output CSV filename
    """
    df = pd.DataFrame({'l': l, 'TT': TT})
    df.to_csv(filename, index=False)
    # Print summary
    pd.set_option("display.precision", 6)
    pd.set_option("display.max_rows", 6)
    print("First and last few rows of the computed CMB TT power spectrum (μK^2):")
    print(df.head(3))
    print("...")
    print(df.tail(3))
    print("\nSaved CMB TT power spectrum to " + filename)

if __name__ == "__main__":
    l, TT = compute_cmb_tt_spectrum()
    save_spectrum_to_csv(l, TT, "data/result.csv")
