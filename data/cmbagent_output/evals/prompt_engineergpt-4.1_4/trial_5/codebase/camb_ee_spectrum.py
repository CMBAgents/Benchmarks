# filename: codebase/camb_ee_spectrum.py
import os
import numpy as np
import pandas as pd

import camb
from camb import model, initialpower

def compute_cmb_ee_spectrum():
    r"""
    Computes the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for a flat Lambda CDM cosmology
    using CAMB, with specified cosmological parameters.

    Returns
    -------
    lvals : ndarray
        Multipole moments (l), shape (N,), units: dimensionless
    cl_ee : ndarray
        E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)), shape (N,), units: micro-Kelvin^2
    """
    # Set cosmological parameters
    H0 = 67.5  # Hubble constant, units: km/s/Mpc
    ombh2 = 0.022  # Omega_b h^2, units: dimensionless
    omch2 = 0.122  # Omega_c h^2, units: dimensionless
    mnu = 0.06  # sum m_nu, units: eV
    omk = 0.0  # Omega_k, units: dimensionless
    tau = 0.04  # Optical depth to reionization, units: dimensionless
    As = 2e-9  # Scalar amplitude, units: dimensionless
    ns = 0.965  # Scalar spectral index, units: dimensionless

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(3000, lens_potential_accuracy=0)
    pars.WantTensors = False

    # Calculate results for these parameters
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000)
    cl = powers['total']  # shape: (lmax+1, 4) for TT, EE, BB, TE

    # l values: 0 to lmax
    lvals = np.arange(cl.shape[0])  # l=0 to lmax
    # Extract EE spectrum (column 1)
    cl_ee_raw = cl[:,1]  # units: micro-Kelvin^2

    # Compute l(l+1)C_l^{EE}/(2pi) for l=2 to l=3000
    lmin = 2
    lmax = 3000
    lvals = lvals[lmin:lmax+1]
    cl_ee = cl_ee_raw[lmin:lmax+1]
    # l(l+1)C_l/(2pi)
    cl_ee = lvals * (lvals + 1) * cl_ee / (2.0 * np.pi)  # units: micro-Kelvin^2

    return lvals, cl_ee

def save_to_csv(lvals, cl_ee, filename):
    r"""
    Save the multipole moments and E-mode power spectrum to a CSV file.

    Parameters
    ----------
    lvals : ndarray
        Multipole moments (l), shape (N,), units: dimensionless
    cl_ee : ndarray
        E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)), shape (N,), units: micro-Kelvin^2
    filename : str
        Output CSV filename
    """
    df = pd.DataFrame({'l': lvals.astype(int), 'EE': cl_ee})
    df.to_csv(filename, index=False)
    print("Saved E-mode polarization power spectrum to " + filename)
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())
    print("Total number of multipoles: " + str(len(df)))


if __name__ == "__main__":
    # Ensure data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    lvals, cl_ee = compute_cmb_ee_spectrum()
    output_file = os.path.join(data_dir, "result.csv")
    save_to_csv(lvals, cl_ee, output_file)
