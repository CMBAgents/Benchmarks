# filename: codebase/calculate_cmb_power_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def calculate_and_save_cmb_power_spectrum():
    r"""
    Calculates the CMB raw temperature power spectrum (C_l^TT) for a specific
    Lambda CDM cosmology using CAMB and saves it to a CSV file.

    The cosmological parameters are:
    - Hubble constant (H0): 74 km/s/Mpc
    - Baryon density (ombh2): 0.022
    - Cold dark matter density (omch2): 0.122
    - Neutrino mass sum (mnu): 0.06 eV
    - Curvature (omk): 0
    - Optical depth to reionization (tau): 0.06
    - Scalar amplitude (As): 2e-9
    - Scalar spectral index (ns): 0.965

    The C_l^TT is computed in units of muK^2 for multipole moments
    l = 2 to l = 3000.
    The results are saved in 'data/result.csv'.
    """
    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=74, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965)
    
    # Set lmax for calculation, up to 3000
    # lens_potential_accuracy=0 as we are primarily interested in unlensed C_ls
    # For unlensed_scalar_cls, lensing calculation isn't strictly needed.
    pars.set_for_lmax(lmax=3000, lens_potential_accuracy=0)

    # Calculate results
    results = camb.get_results(pars)

    # Get CMB power spectra
    # raw_cl=True returns C_l directly. For TT, units are K^2.
    powers = results.get_cmb_power_spectra(pars, raw_cl=True)

    # Extract unlensed scalar C_l^TT
    # powers['unlensed_scalar'] has shape (nspec, lmax+1)
    # nspec=4 for T,E,B,TE. Index 0 is TT.
    # This is C_l^TT in K^2
    cl_TT_K2_all_l = powers['unlensed_scalar'][0]

    # We need l from 2 to 3000
    ls = np.arange(2, 3001)
    
    # Slice the C_l array for l=2 to l=3000
    # Array is 0-indexed, so index i corresponds to l=i
    cl_TT_K2_slice = cl_TT_K2_all_l[2:3001]

    # Convert C_l^TT from K^2 to muK^2
    # 1 K^2 = (10^6 muK)^2 = 10^12 muK^2
    cl_TT_muK2 = cl_TT_K2_slice * 1e12  # units: muK^2

    # Create a Pandas DataFrame
    df = pd.DataFrame({'l': ls, 'TT': cl_TT_muK2})

    # Create data directory if it doesn't exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save to CSV
    file_path = os.path.join(data_dir, 'result.csv')
    df.to_csv(file_path, index=False)

    print("Successfully calculated CMB power spectrum.")
    print("Results saved to: " + str(file_path))
    print("The CSV file contains two columns: 'l' (multipole moment) and 'TT' (raw temperature power spectrum in muK^2).")
    print("Data covers l from 2 to 3000.")
    print("\nFirst 5 rows of the data:")
    # Configure pandas to display float with more precision if needed for verification
    pd.set_option('display.float_format', '{:.6e}'.format)
    print(df.head())
    print("\nLast 5 rows of the data:")
    print(df.tail())
    
    # Print some summary statistics of the computed TT spectrum
    print("\nSummary statistics for TT (muK^2):")
    print(df['TT'].describe())


if __name__ == '__main__':
    calculate_and_save_cmb_power_spectrum()