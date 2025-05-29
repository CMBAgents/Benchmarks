# filename: codebase/calculate_cmb_power_spectrum.py
import camb
import numpy as np
import pandas as pd
import os


def calculate_cmb_power_spectrum():
    r"""
    Calculates the CMB temperature power spectrum for a non-flat Lambda CDM cosmology.

    The function uses specified cosmological parameters to compute the temperature
    power spectrum D_l^TT = l(l+1)C_l^TT/(2pi) in units of muK^2 for multipole
    moments l from 2 to 3000. The results are saved in a CSV file.

    Cosmological Parameters:
    - Hubble constant (H0): 67.3 km/s/Mpc
    - Baryon density (ombh2): 0.022
    - Cold dark matter density (omch2): 0.122
    - Neutrino mass sum (mnu): 0.06 eV
    - Curvature (omk): 0.05
    - Optical depth to reionization (tau): 0.06
    - Scalar amplitude (As): 2e-9
    - Scalar spectral index (ns): 0.965
    """

    # Set up CAMB parameters
    pars = camb.CAMBparams()

    # Set cosmological parameters
    pars.set_cosmology(H0=67.3, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0.05, tau=0.06)
    
    # Set initial power spectrum parameters
    pars.InitPower.set_params(As=2e-9, ns=0.965)

    # Set lmax for calculation, and lensing accuracy
    # We need spectra up to l=3000
    lmax_calc = 3000
    pars.set_for_lmax(lmax_calc, lens_potential_accuracy=1)

    # Get results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra
    # This returns D_l = l(l+1)C_l/(2pi) in muK^2
    # The array has columns: TT, EE, BB, TE
    # powers[l, 0] is D_l^TT for multipole l
    powers = results.get_lensed_scalar_cls(lmax=lmax_calc)  # powers is (lmax_calc+1, 4)

    # Multipole moments from l=2 to l=3000
    ls = np.arange(2, lmax_calc + 1)

    # Temperature power spectrum D_l^TT = l(l+1)C_l^TT/(2pi)
    # powers[:, 0] contains D_l^TT starting from l=0.
    # So, for l in ls, we need powers[l, 0]
    dl_TT = powers[ls, 0]  # Units: muK^2

    # Create a Pandas DataFrame
    df_results = pd.DataFrame({'l': ls, 'TT': dl_TT})

    # Define the output directory and filename
    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_path = os.path.join(output_dir, 'result.csv')

    # Save to CSV
    df_results.to_csv(file_path, index=False)

    print("CMB temperature power spectrum calculation complete.")
    print("Results saved to: " + str(file_path))
    
    # Print some info about the saved data
    print("\nFirst 5 rows of the result.csv:")
    print(df_results.head())
    print("\nLast 5 rows of the result.csv:")
    print(df_results.tail())
    
    # Print min, max, mean of TT for verification
    print("\nSummary statistics for TT (muK^2):")
    print("Min TT: " + str(df_results['TT'].min()))
    print("Max TT: " + str(df_results['TT'].max()))
    print("Mean TT: " + str(df_results['TT'].mean()))


if __name__ == '__main__':
    # Suppress CAMB output
    # Note: CAMB can be verbose. For production, one might redirect stdout/stderr
    # or use logging if CAMB supports it. For this script, default verbosity is fine.
    
    # Set pandas display options for better console output
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.width', 100) 
    
    calculate_cmb_power_spectrum()