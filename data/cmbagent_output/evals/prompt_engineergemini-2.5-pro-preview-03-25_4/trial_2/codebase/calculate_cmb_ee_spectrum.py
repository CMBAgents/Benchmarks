# filename: codebase/calculate_cmb_ee_spectrum.py
import camb
import numpy as np
import pandas as pd
import os


def calculate_cmb_ee_spectrum():
    r"""
    Calculates the CMB E-mode polarization power spectrum D_l^EE = l(l+1)C_l^EE/(2*pi)
    for a given set of cosmological parameters using CAMB.

    The parameters are:
    - H0: 67.5 km/s/Mpc (Hubble constant)
    - ombh2: 0.022 (Baryon density * h^2)
    - omch2: 0.122 (Cold dark matter density * h^2)
    - mnu: 0.06 eV (Sum of neutrino masses)
    - omk: 0.0 (Curvature parameter, 0 for flat)
    - tau: 0.04 (Optical depth to reionization)
    - As: 2.0e-9 (Scalar amplitude)
    - ns: 0.965 (Scalar spectral index)
    - lmax: 3000 (Maximum multipole moment)

    The results (l, D_l^EE in muK^2) for l from 2 to 3000 are saved to 'data/result.csv'.
    """
    # Define cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Baryon density * h^2
    omch2 = 0.122  # Cold dark matter density * h^2
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature parameter
    tau = 0.04  # Optical depth to reionization
    As = 2.0e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    lmax = 3000 # Maximum multipole moment for calculation

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    
    # Configure CAMB for unlensed scalar spectra up to lmax.
    # lens_potential_accuracy=0 ensures primordial (unlensed) Cls are computed.
    pars.set_for_lmax(lmax, lens_potential_accuracy=0) 
    
    # Specify that we want scalar modes, and not tensor or vector modes.
    pars.WantScalars = True
    pars.WantTensors = False 
    pars.WantVectors = False

    # Get results from CAMB
    print("Running CAMB to calculate power spectra...")
    results = camb.get_results(pars)
    print("CAMB calculation finished.")

    # Get CMB power spectra.
    # We are interested in unlensed scalar spectra.
    # The get_cmb_power_spectra function returns D_l = l(l+1)C_l/(2*pi).
    # With CMB_unit='muK', values are in muK^2.
    # We request only 'unlensed_scalar' to be efficient.
    print("Extracting CMB power spectra...")
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['unlensed_scalar'])
    
    # powers['unlensed_scalar'] is an array where columns are TT, EE, BB, TE.
    # EE spectrum is in the second column (index 1).
    # The array is indexed from l=0 up to lmax.
    unlensed_scalar_cls = powers['unlensed_scalar']
    
    # We need multipole moments from l=2 to l=3000.
    # These are the actual multipole numbers for the CSV.
    l_values_for_csv = np.arange(2, lmax + 1) 
    
    # Extract the EE spectrum values for l from 2 to lmax.
    # In the CAMB output array, row index i corresponds to multipole l=i.
    # So, for l=2, index is 2. For l=lmax, index is lmax.
    ee_spectrum_values = unlensed_scalar_cls[2:lmax+1, 1]  # D_l^EE in muK^2

    # Create a Pandas DataFrame
    df_results = pd.DataFrame({'l': l_values_for_csv, 'EE': ee_spectrum_values})

    # Define the output directory and file path
    output_dir = 'data'
    output_filename = 'result.csv'
    output_file = os.path.join(output_dir, output_filename)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: " + output_dir)

    # Save the DataFrame to a CSV file
    df_results.to_csv(output_file, index=False)

    print("\nCMB E-mode polarization power spectrum calculation complete.")
    print("Results saved to: " + output_file)
    
    # Print a sample of the results to the console
    print("\nFirst 5 rows of the results (l, D_l^EE in muK^2):")
    print(df_results.head(5).to_string())
    
    print("\nLast 5 rows of the results (l, D_l^EE in muK^2):")
    print(df_results.tail(5).to_string())
    
    print("\nShape of the results DataFrame: " + str(df_results.shape))
    # Expected shape: (2999, 2) for lmax=3000, range l=2 to 3000.
    
    print("\nStatistical summary of the 'EE' column (D_l^EE in muK^2):")
    print(df_results['EE'].describe().to_string())


if __name__ == '__main__':
    calculate_cmb_ee_spectrum()