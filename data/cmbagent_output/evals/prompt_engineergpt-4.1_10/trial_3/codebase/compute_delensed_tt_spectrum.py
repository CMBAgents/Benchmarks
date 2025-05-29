# filename: codebase/compute_delensed_tt_spectrum.py
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower
import os

def compute_delensed_tt_spectrum(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density [dimensionless]
    omch2=0.122,            # Cold dark matter density [dimensionless]
    mnu=0.06,               # Neutrino mass sum [eV]
    omk=0.0,                # Curvature [dimensionless]
    tau=0.06,               # Optical depth to reionization [dimensionless]
    As=2e-9,                # Scalar amplitude [dimensionless]
    ns=0.965,               # Scalar spectral index [dimensionless]
    lmin=2,                 # Minimum multipole
    lmax=3000,              # Maximum multipole
    delens_efficiency=0.8,  # Delensing efficiency [fraction]
    output_csv='data/result.csv'
):
    r"""
    Compute the delensed CMB temperature power spectrum for a flat Lambda CDM cosmology
    using CAMB, applying a specified delensing efficiency.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Baryon density parameter \Omega_b h^2 (dimensionless).
    omch2 : float
        Cold dark matter density parameter \Omega_c h^2 (dimensionless).
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter \Omega_k (dimensionless).
    tau : float
        Optical depth to reionization (dimensionless).
    As : float
        Scalar amplitude A_s (dimensionless).
    ns : float
        Scalar spectral index n_s (dimensionless).
    lmin : int
        Minimum multipole moment l.
    lmax : int
        Maximum multipole moment l.
    delens_efficiency : float
        Fraction of lensing to be removed (0 = no delensing, 1 = perfect delensing).
    output_csv : str
        Path to output CSV file.

    Returns
    -------
    None
        Saves the delensed spectrum to a CSV file and prints a summary.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.WantTensors = False
    pars.Want_CMB_lensing = True

    # Get results for lensed and unlensed spectra
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    cl_lensed = powers['total'][:, 0]  # TT lensed, units: muK^2
    cl_unlensed = powers['unlensed_scalar'][:, 0]  # TT unlensed, units: muK^2

    # Compute delensed spectrum: C_l^delensed = C_l^unlensed + (1-eff) * (C_l^lensed - C_l^unlensed)
    cl_delensed = cl_unlensed + (1.0 - delens_efficiency) * (cl_lensed - cl_unlensed)

    # Compute l(l+1)C_l/(2pi) for l in [lmin, lmax]
    ells = np.arange(cl_delensed.shape[0])
    valid = (ells >= lmin) & (ells <= lmax)
    ells = ells[valid]
    factor = ells * (ells + 1) / (2.0 * np.pi)
    tt_delensed = factor * cl_delensed[valid]  # units: muK^2

    # Save to CSV
    df = pd.DataFrame({'l': ells, 'TT': tt_delensed})
    df.to_csv(output_csv, index=False)

    # Print summary
    np.set_printoptions(precision=6, suppress=True)
    print("Delensed CMB TT power spectrum (l(l+1)C_l^{TT}/(2pi)) in muK^2 saved to " + output_csv)
    print("First 5 rows:\n", df.head())
    print("Last 5 rows:\n", df.tail())


if __name__ == "__main__":
    compute_delensed_tt_spectrum()