# filename: codebase/camb_cmb_tt.py
import os
import numpy as np
import pandas as pd

# Ensure data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

import camb
from camb import model, initialpower

def compute_cmb_tt_spectrum():
    """
    Computes the CMB temperature power spectrum for the specified flat Lambda CDM cosmology.

    Returns:
        l (ndarray): Multipole moments (2 to 3000).
        TT (ndarray): Temperature power spectrum (l(l+1)C_l^{TT}/(2pi)) in micro-Kelvin^2.
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.02  # Omega_b h^2
    omch2 = 0.122  # Omega_c h^2
    mnu = 0.06  # Sum of neutrino masses [eV]
    omk = 0.0  # Curvature (flat)
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index

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

    # l values start from 2
    l = np.arange(totCL.shape[0])
    # Compute l(l+1)C_l^{TT}/(2pi) in micro-Kelvin^2
    TT = l * (l + 1) * totCL[:, 0] / (2 * np.pi)

    # Select l=2 to l=3000
    l_out = l[2:3001]
    TT_out = TT[2:3001]

    return l_out, TT_out

def save_to_csv(l, TT, filename):
    """
    Saves the multipole moments and TT power spectrum to a CSV file.

    Args:
        l (ndarray): Multipole moments.
        TT (ndarray): Temperature power spectrum in micro-Kelvin^2.
        filename (str): Output CSV filename.
    """
    df = pd.DataFrame({'l': l, 'TT': TT})
    df.to_csv(filename, index=False)
    # Print summary
    pd.set_option("display.precision", 6)
    pd.set_option("display.max_rows", 10)
    print("CMB TT power spectrum (l=2 to l=3000) saved to " + filename)
    print("First and last 5 rows of the result:")
    print(pd.concat([df.head(5), df.tail(5)]))

if __name__ == "__main__":
    l, TT = compute_cmb_tt_spectrum()
    save_to_csv(l, TT, "data/result.csv")
