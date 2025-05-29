# filename: codebase/cmb_power_spectrum.py
import os
import numpy as np
import pandas as pd
try:
    import camb
except ImportError:
    print("CAMB is not installed. Please install it using 'pip install camb'.")
    # In a real multi-agent setup, this script would terminate, 
    # and another agent would handle the installation.
    # For standalone execution, one might raise the error or exit.
    raise

def calculate_cmb_power_spectrum():
    r"""
    Calculates the raw CMB temperature power spectrum (C_l^TT) for a flat Lambda CDM cosmology.

    The function uses specified cosmological parameters to compute the unlensed scalar
    temperature power spectrum C_l^TT in units of muK^2 for multipole moments
    l from 2 to 3000.

    Cosmological Parameters:
        H_0: 70 km/s/Mpc (Hubble constant)
        Omega_b h^2: 0.022 (Baryon density)
        Omega_c h^2: 0.122 (Cold dark matter density)
        Sigma m_nu: 0.06 eV (Neutrino mass sum)
        Omega_k: 0 (Curvature)
        tau: 0.06 (Optical depth to reionization)
        A_s: 2e-9 (Scalar amplitude)
        n_s: 0.965 (Scalar spectral index)

    Returns:
        pandas.DataFrame: A DataFrame with two columns:
            'l': Multipole moment (integer values from 2 to 3000)
            'TT': Temperature power spectrum (C_l^TT in muK^2)
    """
    # Initialize CAMB parameters
    pars = camb.CAMBparams()

    # Set cosmological parameters
    pars.set_cosmology(
        H0=70,
        ombh2=0.022,
        omch2=0.122,
        mnu=0.06,
        omk=0,
        tau=0.06
    )

    # Set initial power spectrum parameters
    pars.InitPower.set_params(
        As=2e-9,
        ns=0.965
    )

    # Set calculation options for unlensed scalar spectra up to lmax=3000
    lmax_calc = 3000
    pars.WantScalars = True
    pars.WantTensors = False  # Not requested, default is False
    pars.DoLensing = False    # For "raw" unlensed spectrum
    pars.set_for_lmax(lmax=lmax_calc, lens_potential_accuracy=0)  # lens_potential_accuracy is not used if DoLensing=False

    # Get results from CAMB
    results = camb.get_results(pars)

    # Get CMB power spectra.
    # With raw_cl=True and CMB_unit='muK', this returns C_l in muK^2.
    # 'unlensed_scalar' contains TT, EE, BB, TE. We need TT (index 0).
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
    cl_TT_unlensed = powers['unlensed_scalar'][:, 0]  # Units: muK^2

    # Multipole moments from l=2 to l=3000
    ls = np.arange(2, lmax_calc + 1)  # l values

    # Extract C_l^TT for l >= 2
    # cl_TT_unlensed is indexed from l=0. So cl_TT_unlensed[l] is C_l.
    cl_TT_values = cl_TT_unlensed[ls]

    # Create DataFrame
    df = pd.DataFrame({'l': ls, 'TT': cl_TT_values})
    
    return df

def main():
    r"""
    Main function to execute the CMB power spectrum calculation,
    save the results to a CSV file, and print a summary.
    """
    # Ensure the data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # Calculate power spectrum
    try:
        cmb_df = calculate_cmb_power_spectrum()
    except Exception as e:
        print("An error occurred during CAMB calculation:")
        print(str(e))
        return

    # Save results to CSV
    output_filename = 'data/result.csv'
    try:
        cmb_df.to_csv(output_filename, index=False)
        print("CMB TT power spectrum calculated and saved to " + output_filename)
    except Exception as e:
        print("An error occurred while saving the CSV file:")
        print(str(e))
        return

    # Print selected values from the DataFrame
    print("\nSelected C_l^TT values (in muK^2):")
    selected_ls = [2, 20, 200, 1000, 2000, 3000]
    # Filter DataFrame for selected l values
    # Ensure selected_ls are within the calculated range
    min_l_calc = cmb_df['l'].min()
    max_l_calc = cmb_df['l'].max()
    valid_selected_ls = [l_val for l_val in selected_ls if min_l_calc <= l_val <= max_l_calc]
    
    if valid_selected_ls:
        selected_values_df = cmb_df[cmb_df['l'].isin(valid_selected_ls)]
        # Set pandas options for printing full content
        pd.set_option('display.max_rows', len(selected_values_df))
        pd.set_option('display.width', 1000)  # Adjust width as needed
        pd.set_option('display.max_colwidth', None)  # Display full column content
        print(selected_values_df.to_string(index=False))
    else:
        print("No data available for the selected l values.")
        
    # Print first 5 and last 5 values as a sample
    print("\nFirst 5 rows of the result:")
    print(cmb_df.head().to_string())
    print("\nLast 5 rows of the result:")
    print(cmb_df.tail().to_string())


if __name__ == '__main__':
    main()