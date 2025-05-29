# filename: codebase/compute_delensed_cmb_tt_spectrum.py
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower
import os


def compute_delensed_cmb_tt_spectrum(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density [dimensionless]
    omch2=0.122,            # Cold dark matter density [dimensionless]
    mnu=0.06,               # Neutrino mass sum [eV]
    omk=0,                  # Curvature [dimensionless]
    tau=0.06,               # Optical depth to reionization [dimensionless]
    As=2e-9,                # Scalar amplitude [dimensionless]
    ns=0.965,               # Scalar spectral index [dimensionless]
    lmin=2,                 # Minimum multipole
    lmax=3000,              # Maximum multipole
    delensing_efficiency=0.8, # Delensing efficiency [fraction]
    output_csv='data/result.csv' # Output CSV file
):
    r"""
    Compute the delensed CMB temperature power spectrum for a flat Lambda CDM cosmology.

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
    delensing_efficiency : float
        Fraction of lensing removed (0 = no delensing, 1 = perfect delensing).
    output_csv : str
        Path to output CSV file.

    Returns
    -------
    None
        Saves the delensed spectrum to a CSV file and prints summary to console.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up CAMB parameters for lensed spectrum
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.WantTensors = False
    pars.Want_CMB_lensing = True

    # Compute lensed CMB power spectra
    results_lensed = camb.get_results(pars)
    powers_lensed = results_lensed.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total'])
    cl_lensed = powers_lensed['total'][:, 0]  # TT spectrum, units: muK^2

    # Set up CAMB parameters for unlensed spectrum
    pars.Want_CMB_lensing = False
    results_unlensed = camb.get_results(pars)
    powers_unlensed = results_unlensed.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['unlensed_scalar'])
    cl_unlensed = powers_unlensed['unlensed_scalar'][:, 0]  # TT spectrum, units: muK^2

    # Compute delensed spectrum
    # delensed = unlensed + (lensed - unlensed) * (1 - efficiency)
    cl_delensed = cl_unlensed + (cl_lensed - cl_unlensed) * (1.0 - delensing_efficiency)

    # Compute l(l+1)C_l/(2pi) for l=2 to lmax
    ells = np.arange(lmin, lmax + 1)
    factor = ells * (ells + 1) / (2.0 * np.pi)
    # CAMB returns Cl array starting at l=0, so index accordingly
    cl_delensed_plot = cl_delensed[lmin:lmax + 1]
    tt_spectrum = factor * cl_delensed_plot  # units: muK^2

    # Save to CSV
    df = pd.DataFrame({'l': ells, 'TT': tt_spectrum})
    df.to_csv(output_csv, index=False)

    # Print summary to console
    print("Delensed CMB TT power spectrum (l(l+1)C_l^{TT}/(2pi)) saved to " + output_csv)
    print("Columns: l (multipole), TT (muK^2)")
    print("First 10 rows:")
    print(df.head(10).to_string(index=False))
    print("\nLast 10 rows:")
    print(df.tail(10).to_string(index=False))
    print("\nTotal number of multipoles: " + str(len(df)))


if __name__ == "__main__":
    compute_delensed_cmb_tt_spectrum()