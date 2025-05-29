# filename: codebase/delensed_cmb_spectrum.py
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
    output_csv='data/result.csv'
):
    r"""
    Compute the delensed CMB temperature power spectrum l(l+1)C_l^{TT}/(2pi) in μK^2
    for a flat Lambda CDM cosmology using CAMB, applying a specified delensing efficiency.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter (Ω_b h^2).
    omch2 : float
        Physical cold dark matter density parameter (Ω_c h^2).
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter (Ω_k).
    tau : float
        Optical depth to reionization.
    As : float
        Scalar amplitude.
    ns : float
        Scalar spectral index.
    lmin : int
        Minimum multipole moment (inclusive).
    lmax : int
        Maximum multipole moment (inclusive).
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

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.WantTensors = False

    # Get lensed CMB power spectra
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total', 'unlensed_scalar'])
    lensed_cl = powers['total'][:, 0]  # TT, lensed
    unlensed_cl = powers['unlensed_scalar'][:, 0]  # TT, unlensed

    # Multipole array
    ell = np.arange(powers['total'].shape[0])

    # Only keep l in [lmin, lmax]
    mask = (ell >= lmin) & (ell <= lmax)
    ell = ell[mask]
    lensed_cl = lensed_cl[mask]
    unlensed_cl = unlensed_cl[mask]

    # Compute delensed spectrum
    # delensed = unlensed + (lensed - unlensed) * (1 - delensing_efficiency)
    delensed_cl = unlensed_cl + (lensed_cl - unlensed_cl) * (1.0 - delensing_efficiency)

    # Compute l(l+1)C_l/(2pi) in μK^2
    factor = ell * (ell + 1) / (2.0 * np.pi)
    delensed_tt = factor * delensed_cl  # μK^2

    # Save to CSV
    df = pd.DataFrame({'l': ell, 'TT': delensed_tt})
    df.to_csv(output_csv, index=False)

    # Print summary to console
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print("Delensed CMB TT power spectrum (l(l+1)C_l^{TT}/(2pi)) in μK^2, l = " + str(lmin) + " to " + str(lmax))
    print("First 10 rows:")
    print(df.head(10).to_string(index=False))
    print("\nSaved full results to " + output_csv)


if __name__ == "__main__":
    compute_delensed_cmb_tt_spectrum()