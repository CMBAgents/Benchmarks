# filename: codebase/cmb_power_spectrum.py
r"""
Compute the CMB temperature power spectrum for a non-flat Lambda CDM cosmology using CAMB.

Parameters:
    H0 (float): Hubble constant in km/s/Mpc [unit: km/s/Mpc]
    ombh2 (float): Baryon density parameter [unitless]
    omch2 (float): Cold dark matter density parameter [unitless]
    mnu (float): Sum of neutrino masses [unit: eV]
    omk (float): Curvature parameter [unitless]
    tau (float): Optical depth to reionization [unitless]
    As (float): Scalar amplitude [unitless]
    ns (float): Scalar spectral index [unitless]

Output:
    CSV file 'data/result.csv' with columns:
        l: Multipole moment (integer, 2 to 3000)
        TT: Temperature power spectrum l(l+1)C_l^{TT}/(2pi) [unit: μK^2]
"""

import os
import numpy as np
import pandas as pd

# Ensure output directory exists
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    import camb
    from camb import model, initialpower
except ImportError as e:
    print("CAMB is not installed. Please install camb and rerun the code.")
    raise e


def compute_cmb_tt_spectrum():
    r"""
    Compute the CMB temperature power spectrum for the specified non-flat Lambda CDM cosmology.

    Returns:
        l (ndarray): Multipole moments [unitless]
        TT (ndarray): Temperature power spectrum l(l+1)C_l^{TT}/(2pi) [unit: μK^2]
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

    # l values: CAMB returns from l=0, so index accordingly
    l = np.arange(totCL.shape[0])
    lmin = 2
    lmax = 3000
    l = l[lmin:lmax+1]
    # TT spectrum: column 0 is TT
    TT = totCL[lmin:lmax+1, 0]  # [μK^2]

    # Compute l(l+1)C_l^{TT}/(2pi) [μK^2]
    TT_power = l * (l + 1) * TT / (2.0 * np.pi)

    return l, TT_power


def save_to_csv(l, TT, filename):
    r"""
    Save the multipole moments and TT power spectrum to a CSV file.

    Args:
        l (ndarray): Multipole moments [unitless]
        TT (ndarray): Temperature power spectrum [μK^2]
        filename (str): Output CSV file path
    """
    df = pd.DataFrame({'l': l, 'TT': TT})
    df.to_csv(filename, index=False)
    print("Saved CMB temperature power spectrum to " + filename)
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())
    print("Total number of multipoles: " + str(len(df)))


if __name__ == "__main__":
    l, TT = compute_cmb_tt_spectrum()
    output_file = os.path.join("data", "result.csv")
    save_to_csv(l, TT, output_file)
