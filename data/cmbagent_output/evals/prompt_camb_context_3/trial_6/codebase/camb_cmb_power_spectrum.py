# filename: codebase/camb_cmb_power_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_tt_spectrum():
    r"""
    Compute the CMB raw temperature power spectrum (C_l^{TT}) for a flat Lambda CDM cosmology
    using CAMB with the following parameters:
        - Hubble constant (H0): 74 km/s/Mpc
        - Baryon density (Omega_b h^2): 0.022
        - Cold dark matter density (Omega_c h^2): 0.122
        - Neutrino mass sum (Sigma m_nu): 0.06 eV
        - Curvature (Omega_k): 0
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (A_s): 2e-9
        - Scalar spectral index (n_s): 0.965

    The function computes the lensed scalar temperature power spectrum (C_l^{TT}) in units of muK^2
    for multipole moments l=2 to l=3000, and saves the results to 'data/result.csv' with columns:
        - l: Multipole moment (integer values from 2 to 3000)
        - TT: Raw temperature power spectrum (muK^2)

    Returns
    -------
    None
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=74, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965)
    pars.set_for_lmax(lmax=3000)
    pars.DoLensing = True

    # Compute results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra (raw Cl, in muK^2)
    powers = results.get_cmb_power_spectra(params=pars, CMB_unit='muK', raw_cl=True, spectra=['lensed_scalar'])
    cl_tt = powers['lensed_scalar'][:, 0]  # TT is the first column

    # Prepare l and TT arrays for l=2..3000
    l_vals = np.arange(2, 3001)
    tt_vals = cl_tt[2:3001]

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'l': l_vals, 'TT': tt_vals})
    df.to_csv(output_path, index=False)

    # Print summary to console
    np.set_printoptions(precision=6, suppress=True)
    pd.set_option("display.precision", 6)
    pd.set_option("display.max_rows", 10)
    print("CMB raw temperature power spectrum (C_l^{TT}) computed for l=2 to l=3000.")
    print("Results saved to " + output_path)
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())
    print("TT units: microkelvin^2 (muK^2)")
    print("l: Multipole moment (dimensionless)")

if __name__ == "__main__":
    compute_cmb_tt_spectrum()