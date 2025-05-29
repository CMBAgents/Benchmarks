# filename: codebase/camb_cmb_power_spectrum.py
import os
import numpy as np
import pandas as pd
import camb

def calculate_cmb_power_spectrum():
    r"""
    Calculates the raw Cosmic Microwave Background (CMB) temperature power spectrum (C_l^TT)
    for a flat Lambda CDM cosmology using specified parameters with CAMB.

    The cosmological parameters used are:
    - Hubble constant (H_0): 70 km/s/Mpc
    - Baryon density (Omega_b h^2): 0.022
    - Cold dark matter density (Omega_c h^2): 0.122
    - Neutrino mass sum (Sigma m_nu): 0.06 eV
    - Curvature (Omega_k): 0
    - Optical depth to reionization (tau): 0.06
    - Scalar amplitude (A_s): 2e-9
    - Scalar spectral index (n_s): 0.965

    The temperature power spectrum (C_l^TT) is computed in units of microK^2
    for multipole moments l from 2 to 3000.

    The results are saved in a CSV file named 'result.csv' in the 'data/' directory,
    with two columns: 'l' (multipole moment) and 'TT' (C_l^TT in microK^2).
    """
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Created directory: " + data_dir)
    else:
        print("Directory " + data_dir + " already exists.")

    # Set cosmological parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=70, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965)
    
    # Set lmax and specify unlensed raw Cls.
    # lens_potential_accuracy=0 means no lensing calculations.
    # Raw Cls from get_cmb_power_spectra with raw_cl=True are unlensed.
    pars.set_for_lmax(lmax=3000, lens_potential_accuracy=0)
    
    # We want scalar spectra, not tensor
    pars.WantTensors = False 
    
    # Calculate results
    print("Running CAMB to calculate cosmological results...")
    results = camb.get_results(pars)
    print("CAMB calculation complete.")
    
    # Get raw C_l values in K^2.
    # raw_cl=True ensures C_l values are returned.
    # CMB_unit='K' ensures the units are K^2.
    print("Extracting CMB power spectra...")
    powers = results.get_cmb_power_spectra(pars, CMB_unit='K', raw_cl=True)
    
    # powers is a dictionary. 'total' contains TT, EE, BB, TE.
    # cls is a numpy array with columns: l, C_l^TT, C_l^EE, C_l^BB, C_l^TE
    cls = powers['total']
    
    # Extract l and C_l^TT
    # ls from CAMB are floats but represent integers. Convert to int.
    ls = cls[:,0].astype(int)
    cl_TT_K2 = cls[:,1]  # These are C_l^TT in K^2
    
    # Filter for l from 2 to 3000
    # CAMB output for raw_cl=True starts l from 0.
    # C_l for l=0 and l=1 are typically zero for TT from scalar perturbations.
    mask = (ls >= 2) & (ls <= 3000)
    ls_filtered = ls[mask]
    cl_TT_K2_filtered = cl_TT_K2[mask]
    
    # Convert C_l^TT from K^2 to microK^2
    # 1 K = 10^6 microK, so 1 K^2 = (10^6 microK)^2 = 1e12 microK^2
    cl_TT_muK2 = cl_TT_K2_filtered * 1e12
    
    # Create pandas DataFrame
    df_results = pd.DataFrame({'l': ls_filtered, 'TT': cl_TT_muK2})
    
    # Save to CSV
    file_path = os.path.join(data_dir, "result.csv")
    df_results.to_csv(file_path, index=False)
    
    print("CMB temperature power spectrum calculated and saved to " + file_path)
    
    # Set pandas display options for better console output
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', None)

    print("\nPreview of the first few rows of the data:")
    print(df_results.head())
    print("\nPreview of the last few rows of the data:")
    print(df_results.tail())
    print("\nSummary of the data:")
    print("Number of l values: " + str(len(df_results)))
    if len(df_results) > 0:
        print("Min l: " + str(df_results['l'].min()))
        print("Max l: " + str(df_results['l'].max()))
        print("Mean TT (microK^2): " + str(df_results['TT'].mean()))
        print("Min TT (microK^2): " + str(df_results['TT'].min()))
        print("Max TT (microK^2): " + str(df_results['TT'].max()))
    else:
        print("No data to summarize.")


if __name__ == '__main__':
    calculate_cmb_power_spectrum()