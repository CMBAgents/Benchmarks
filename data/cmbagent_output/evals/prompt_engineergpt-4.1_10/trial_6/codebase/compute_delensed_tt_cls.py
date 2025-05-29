# filename: codebase/compute_delensed_tt_cls.py
import numpy as np
import pandas as pd
import os
import time

from camb import model, get_results, CAMBparams


def compute_delensed_tt_cls(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density [dimensionless]
    omch2=0.122,            # Cold dark matter density [dimensionless]
    mnu=0.06,               # Neutrino mass sum [eV]
    omk=0.0,                # Curvature [dimensionless]
    tau=0.06,               # Optical depth to reionization [dimensionless]
    As=2e-9,                # Scalar amplitude [dimensionless]
    ns=0.965,               # Scalar spectral index [dimensionless]
    lmax=3000,              # Maximum multipole
    delensing_efficiency=0.8, # Delensing efficiency [fraction]
    output_csv="data/result.csv"
):
    r"""
    Compute the delensed CMB temperature power spectrum l(l+1)C_l^{TT}/(2\pi) in units of microK^2
    for a flat Lambda CDM cosmology using CAMB, applying a specified delensing efficiency.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Baryon density parameter Omega_b h^2.
    omch2 : float
        Cold dark matter density parameter Omega_c h^2.
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter Omega_k.
    tau : float
        Optical depth to reionization.
    As : float
        Scalar amplitude.
    ns : float
        Scalar spectral index.
    lmax : int
        Maximum multipole moment l.
    delensing_efficiency : float
        Fraction of lensing to be removed (0 = no delensing, 1 = fully delensed).
    output_csv : str
        Path to output CSV file.

    Returns
    -------
    None
        Saves the result to a CSV file and prints a summary.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up CAMB parameters
    pars = CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.WantTensors = False

    # Get lensed results
    results = get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=False)
    lensed_cls = powers['total'][:, 0]  # TT, lensed

    # Get unlensed results
    powers_unlensed = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=False, lens_potential=False)
    unlensed_cls = powers_unlensed['unlensed_scalar'][:, 0]  # TT, unlensed

    # Multipole array
    ell = np.arange(powers['total'].shape[0])  # l = 0, 1, ..., lmax

    # Only keep l=2 to lmax
    lmin = 2
    mask = (ell >= lmin) & (ell <= lmax)
    ell = ell[mask]
    lensed_cls = lensed_cls[mask]
    unlensed_cls = unlensed_cls[mask]

    # Compute delensed spectrum
    # delensed = unlensed + (lensed - unlensed) * (1 - efficiency)
    delensed_cls = unlensed_cls + (lensed_cls - unlensed_cls) * (1.0 - delensing_efficiency)

    # Compute l(l+1)C_l/(2pi) in muK^2
    factor = ell * (ell + 1) / (2.0 * np.pi)
    delensed_tt = factor * delensed_cls  # [muK^2]

    # Save to CSV
    df = pd.DataFrame({'l': ell.astype(int), 'TT': delensed_tt})
    df.to_csv(output_csv, index=False)

    # Print summary
    print("Delensed CMB TT power spectrum computed and saved to " + output_csv)
    print("Columns: l (multipole), TT (l(l+1)C_l^{TT}/(2pi) in microK^2)")
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())
    print("Total number of multipoles: " + str(len(df)))


if __name__ == "__main__":
    compute_delensed_tt_cls()