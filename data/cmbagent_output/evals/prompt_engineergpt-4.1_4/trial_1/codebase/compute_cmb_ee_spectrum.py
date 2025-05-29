# filename: codebase/compute_cmb_ee_spectrum.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

# Ensure output directory exists
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def compute_cmb_ee_spectrum():
    r"""
    Compute the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for the specified cosmology.

    Returns
    -------
    l : ndarray
        Multipole moments (l), shape (N,)
    EE : ndarray
        E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) in micro-Kelvin squared (uK^2), shape (N,)
    """
    # Set cosmological parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.04)
    pars.InitPower.set_params(As=2e-9, ns=0.965)
    pars.set_for_lmax(3000, lens_potential_accuracy=0)
    pars.WantTensors = False

    # Calculate results for these parameters
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000)
    # 'total' includes l=0,1,2,..., so index l=2 is at index 2
    EE = powers['total'][:,2]  # EE spectrum in muK^2

    # l array: powers['total'] has shape (lmax+1, 4), l runs from 0 to lmax
    l = np.arange(powers['total'].shape[0])

    # Compute l(l+1)C_l^{EE}/(2pi) for l=2 to l=3000
    lmin = 2
    lmax = 3000
    l = l[lmin:lmax+1]
    EE = EE[lmin:lmax+1]
    # Convert to l(l+1)C_l/(2pi)
    EE = l * (l + 1) * EE / (2.0 * np.pi)  # units: muK^2

    return l, EE


def save_results_to_csv(l, EE, filename):
    r"""
    Save the multipole moments and E-mode power spectrum to a CSV file.

    Parameters
    ----------
    l : ndarray
        Multipole moments (l)
    EE : ndarray
        E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) in micro-Kelvin squared (uK^2)
    filename : str
        Output CSV filename
    """
    df = pd.DataFrame({'l': l.astype(int), 'EE': EE})
    df.to_csv(filename, index=False)
    print("Saved E-mode polarization power spectrum to " + filename)
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())


if __name__ == "__main__":
    l, EE = compute_cmb_ee_spectrum()
    save_results_to_csv(l, EE, os.path.join("data", "result.csv"))
