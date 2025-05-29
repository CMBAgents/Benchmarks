# filename: codebase/compute_delensed_CMB.py
r"""
Compute the delensed CMB temperature power spectrum (l(l+1)C_l^{TT}/(2pi)) for a flat Lambda CDM cosmology
using CAMB, with specified cosmological parameters and 80% delensing efficiency.
Save the results in data/result.csv with columns: l, TT (in microK^2).
"""

import numpy as np
import pandas as pd
import camb
from camb import model, initialpower
import os


def compute_delensed_tt_spectrum(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density Omega_b h^2
    omch2=0.122,            # Cold dark matter density Omega_c h^2
    mnu=0.06,               # Neutrino mass sum [eV]
    omk=0,                  # Curvature Omega_k
    tau=0.06,               # Optical depth to reionization
    As=2e-9,                # Scalar amplitude
    ns=0.965,               # Scalar spectral index
    lmin=2,                 # Minimum multipole
    lmax=3000,              # Maximum multipole
    delensing_efficiency=0.8, # Delensing efficiency (fraction of lensing removed)
    output_csv="data/result.csv"
):
    r"""
    Compute the delensed CMB temperature power spectrum and save to CSV.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Baryon density Omega_b h^2.
    omch2 : float
        Cold dark matter density Omega_c h^2.
    mnu : float
        Neutrino mass sum in eV.
    omk : float
        Curvature Omega_k.
    tau : float
        Optical depth to reionization.
    As : float
        Scalar amplitude.
    ns : float
        Scalar spectral index.
    lmin : int
        Minimum multipole.
    lmax : int
        Maximum multipole.
    delensing_efficiency : float
        Fraction of lensing removed (0 = fully lensed, 1 = fully unlensed).
    output_csv : str
        Path to output CSV file.
    """
    # Ensure output directory exists
    outdir = os.path.dirname(output_csv)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.WantTensors = False

    # Compute lensed and unlensed C_ell
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=False)
    cl_lensed = powers['total'][:,0]   # lensed TT, units: muK^2
    cl_unlensed = powers['unlensed_scalar'][:,0] # unlensed TT, units: muK^2

    # l values from 0 to lmax
    ell = np.arange(cl_lensed.size)
    # Restrict to lmin <= l <= lmax
    mask = (ell >= lmin) & (ell <= lmax)
    ell = ell[mask]
    cl_lensed = cl_lensed[mask]
    cl_unlensed = cl_unlensed[mask]

    # Compute delensed spectrum
    # delensed = unlensed + (lensed - unlensed) * (1 - efficiency)
    delensed_cl = cl_unlensed + (cl_lensed - cl_unlensed) * (1.0 - delensing_efficiency)

    # Compute l(l+1)C_l/(2pi)
    factor = ell * (ell + 1) / (2.0 * np.pi)
    TT = factor * delensed_cl  # units: muK^2

    # Save to CSV
    df = pd.DataFrame({'l': ell, 'TT': TT})
    df.to_csv(output_csv, index=False)

    # Print results to console
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    pd.set_option("display.max_rows", 20)
    pd.set_option("display.max_columns", 2)
    pd.set_option("display.width", 120)
    print("Delensed CMB temperature power spectrum (l(l+1)C_l^{TT}/(2pi)) in microK^2, l from " + str(lmin) + " to " + str(lmax) + ":")
    print(df.head(10))
    print("...")
    print(df.tail(10))
    print("\nSaved delensed spectrum to " + output_csv)


if __name__ == "__main__":
    compute_delensed_tt_spectrum()