# filename: codebase/camb_cmb_tt_spectrum.py
import os
import numpy as np
import pandas as pd

# Ensure data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

try:
    import camb
    from camb import model, initialpower
except ImportError as e:
    print("CAMB package is required but not installed. Please install camb and rerun the code.")
    raise e

def compute_cmb_tt_spectrum():
    r"""
    Computes the CMB temperature power spectrum (C_l^{TT}) for the specified flat Lambda CDM cosmology.

    Returns:
        l_vals (np.ndarray): Multipole moments (l) [unitless]
        cl_tt (np.ndarray): Temperature power spectrum (C_l^{TT}) in micro-Kelvin^2 [unit: uK^2]
    """
    # Cosmological parameters
    H0 = 70.0  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [unitless]
    omch2 = 0.122  # Cold dark matter density [unitless]
    mnu = 0.06  # Neutrino mass sum [eV]
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

    # Calculate results for these parameters
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000)
    totCL = powers['total']  # Shape: (lmax+1, 4): TT, EE, BB, TE

    # l values: CAMB returns from l=0, but we want l=2 to l=3000
    l_vals = np.arange(2, 3001)  # [unitless]
    cl_tt = totCL[2:3001, 0]  # TT spectrum in muK^2 [unit: uK^2]

    return l_vals, cl_tt

def save_spectrum_to_csv(l_vals, cl_tt, filename):
    r"""
    Saves the multipole moments and TT power spectrum to a CSV file.

    Args:
        l_vals (np.ndarray): Multipole moments (l) [unitless]
        cl_tt (np.ndarray): Temperature power spectrum (C_l^{TT}) in micro-Kelvin^2 [unit: uK^2]
        filename (str): Output CSV file path
    """
    df = pd.DataFrame({'l': l_vals, 'TT': cl_tt})
    df.to_csv(filename, index=False)
    # Print summary
    print("CMB TT power spectrum saved to " + filename)
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())
    print("l range: " + str(l_vals[0]) + " to " + str(l_vals[-1]))
    print("TT (C_l^{TT}) min: " + str(np.min(cl_tt)) + " uK^2, max: " + str(np.max(cl_tt)) + " uK^2")


if __name__ == "__main__":
    l_vals, cl_tt = compute_cmb_tt_spectrum()
    save_spectrum_to_csv(l_vals, cl_tt, "data/result.csv")
