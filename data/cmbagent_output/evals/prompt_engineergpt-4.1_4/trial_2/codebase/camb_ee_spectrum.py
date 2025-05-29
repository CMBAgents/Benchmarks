# filename: codebase/camb_ee_spectrum.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

def compute_cmb_ee_spectrum():
    r"""
    Compute the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for a flat Lambda CDM cosmology
    using CAMB, with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Neutrino mass sum (mnu): 0.06 eV
        - Curvature (omk): 0
        - Optical depth to reionization (tau): 0.04
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965

    The function computes the E-mode power spectrum (l(l+1)C_l^{EE}/(2pi)) in units of microKelvin^2
    for multipole moments from l=2 to l=3000, and saves the results in a CSV file named 'result.csv'
    with columns:
        - l: Multipole moment (integer values from 2 to 3000)
        - EE: E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi) in microKelvin^2)
    The file is saved in the 'data/' directory.
    """
    # Output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.04  # Optical depth to reionization [dimensionless]
    As = 2e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]

    # Multipole range
    lmin = 2
    lmax = 3000

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.WantTensors = False

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total'])

    # Extract EE spectrum
    cl = powers['total']  # columns: TT, EE, BB, TE, [PP]
    # cl shape: (lmax+1, ncol)
    # cl[:,1] is EE
    ell = np.arange(cl.shape[0])  # l=0,...,lmax
    EE = cl[:,1]  # EE spectrum in muK^2

    # Compute l(l+1)C_l^{EE}/(2pi) for l=2 to lmax
    lvals = np.arange(lmin, lmax+1)
    factor = lvals * (lvals + 1) / (2.0 * np.pi)
    EE_power = factor * EE[lmin:lmax+1]  # units: muK^2

    # Save to CSV
    df = pd.DataFrame({'l': lvals, 'EE': EE_power})
    output_path = os.path.join(output_dir, "result.csv")
    df.to_csv(output_path, index=False)

    # Print summary of results
    pd.set_option("display.precision", 6)
    pd.set_option("display.width", 120)
    print("CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) [muK^2] saved to " + output_path)
    print("First 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())

if __name__ == "__main__":
    compute_cmb_ee_spectrum()