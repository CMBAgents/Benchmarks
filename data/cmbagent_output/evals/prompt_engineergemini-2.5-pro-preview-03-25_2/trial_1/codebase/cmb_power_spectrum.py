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
        H0=70,          # Hubble constant in km/s/Mpc
        ombh2=0.022,    # Physical baryon density
        omch2=0.122,    # Physical cold dark matter density
        mnu=0.06,       # Sum of neutrino masses in eV
        omk=0,          # Curvature parameter
        tau=0.06        # Optical depth to reionization
    )

    # Set initial power spectrum parameters
    pars.InitPower.set_params(
        As=2e-9,        # Scalar amplitude
        ns=0.965        # Scalar spectral index
    )

    # Set calculation options for unlensed scalar spectra up to lmax=3000
    lmax_calc = 3000
    pars.WantScalars = True
    pars.WantTensors = False 
    pars.DoLensing = False   # For "raw" unlensed spectrum
    # AccuracyBoost is more general than lens_potential_accuracy for lmax settings
    pars.set_for_lmax(lmax=lmax_calc, max_eta_k=None, k_eta_max_scalar=None, lens_potential_accuracy=0, AccuracyBoost=1)


    # Get results from CAMB
    results = camb.get_results(pars)

    # Get CMB power spectra.
    # Setting raw_cl=True returns C_l directly.
    # CMB_unit='muK' ensures C_l is in muK^2.
    # 'unlensed_scalar' contains TT, EE, BB, TE. We need TT (index 0).
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
    cl_TT_unlensed = powers['unlensed_scalar'][:, 0]  # Units: muK^2

    # Multipole moments from l=2 to l=3000
    ls = np.arange(2, lmax_calc + 1) # l values

    # Extract C_l^TT for l >= 2
    # cl_TT_unlensed is indexed from l=0. So cl_TT_unlensed[l] is C_l.
    # We need values from l=2 up to lmax_calc.
    # The array from CAMB (cl_TT_unlensed) has length lmax_calc + 1 (indices 0 to lmax_calc)
    # So, to get C_l for l in ls, we slice cl_TT_unlensed from index 2 up to lmax_calc.
    cl_TT_values = cl_TT_unlensed[2:lmax_calc + 1]

    # Create DataFrame
    # Ensure ls and cl_TT_values have the same length
    if len(ls) != len(cl_TT_values):
        # This case should ideally not happen if lmax_calc is handled correctly.
        # If CAMB returns fewer values than lmax_calc for some reason,
        # adjust ls to match the length of cl_TT_values.
        # However, CAMB with raw_cl=True should return up to lmax_calc.
        # The slice cl_TT_unlensed[2:lmax_calc+1] has length lmax_calc - 2 + 1 = lmax_calc -1
        # The ls = np.arange(2, lmax_calc + 1) has length lmax_calc - 2 + 1 = lmax_calc -1
        # So lengths should match.
        # For safety, one could add:
        # max_l_returned = len(cl_TT_unlensed) -1 # Max l for which C_l is returned
        # ls_adjusted = np.arange(2, min(lmax_calc, max_l_returned) + 1)
        # cl_TT_values_adjusted = cl_TT_unlensed[2:min(lmax_calc, max_l_returned) + 1]
        # df = pd.DataFrame({'l': ls_adjusted, 'TT': cl_TT_values_adjusted})
        # But with current CAMB behavior, this is not strictly needed.
        pass # Assuming lengths match based on CAMB's behavior.

    df = pd.DataFrame({'l': ls, 'TT': cl_TT_values})
    
    return df

def main():
    r"""
    Main function to execute the CMB power spectrum calculation,
    save the results to a CSV file, and print a summary.
    """
    # Ensure the data directory exists
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Created directory: " + data_dir)

    # Calculate power spectrum
    try:
        cmb_df = calculate_cmb_power_spectrum()
    except Exception as e:
        print("An error occurred during CAMB calculation:")
        print(str(e))
        # Print traceback for more details if needed for debugging
        # import traceback
        # traceback.print_exc()
        return

    # Save results to CSV
    output_filename = os.path.join(data_dir, 'result.csv')
    try:
        cmb_df.to_csv(output_filename, index=False, float_format='%.8e') # Using scientific notation for precision
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
    if not cmb_df.empty:
        min_l_calc = cmb_df['l'].min()
        max_l_calc = cmb_df['l'].max()
        valid_selected_ls = [l_val for l_val in selected_ls if min_l_calc <= l_val <= max_l_calc]
        
        if valid_selected_ls:
            selected_values_df = cmb_df[cmb_df['l'].isin(valid_selected_ls)].copy() # Use .copy() to avoid SettingWithCopyWarning
            # Format TT column for better readability if needed, though to_string handles it.
            # selected_values_df['TT'] = selected_values_df['TT'].apply(lambda x: "%.6e" % x) 
            
            # Set pandas options for printing full content
            pd.set_option('display.max_rows', None) # Show all rows if selected_values_df is long
            pd.set_option('display.width', 1000) 
            pd.set_option('display.max_colwidth', None) 
            pd.set_option('display.float_format', '{:.6e}'.format) # Format floats in scientific notation

            print(selected_values_df.to_string(index=False))
            
            # Reset pandas options to default if they might affect other parts of a larger application
            pd.reset_option('display.max_rows')
            pd.reset_option('display.width')
            pd.reset_option('display.max_colwidth')
            pd.reset_option('display.float_format')

        else:
            print("No data available for the specified selected l values in the calculated range.")
            
        # Print first 5 and last 5 values as a sample
        print("\nFirst 5 rows of the result:")
        pd.set_option('display.float_format', '{:.6e}'.format)
        print(cmb_df.head().to_string())
        print("\nLast 5 rows of the result:")
        print(cmb_df.tail().to_string())
        pd.reset_option('display.float_format')

    else:
        print("CMB DataFrame is empty. Cannot display selected values or head/tail.")


if __name__ == '__main__':
    main()
