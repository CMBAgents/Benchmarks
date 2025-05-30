# filename: codebase/cmb_power_spectrum.py
# ----------------------------------------------------------------------------------
# CMB Power Spectrum Calculation and Validation using CAMB
# ----------------------------------------------------------------------------------
# Purpose:
# This script calculates the Cosmic Microwave Background (CMB) raw temperature
# power spectrum (C_l^TT) for a flat Lambda CDM cosmology using the CAMB library.
# It saves the results to a CSV file and generates a plot of the power spectrum.
# Additionally, it validates the content of the CSV file.
#
# Cosmological Parameters Used:
# - Hubble constant (H0): 74 km/s/Mpc
# - Baryon density (ombh2): 0.022
# - Cold dark matter density (omch2): 0.122
# - Neutrino mass sum (mnu): 0.06 eV (num_massive_neutrinos=1, standard_neutrino_neff=3.046)
# - Curvature (omk): 0 (flat universe)
# - Optical depth to reionization (tau): 0.06
# - Scalar amplitude (As): 2e-9
# - Scalar spectral index (ns): 0.965
#
# Multipole Range:
# - l = 2 to l = 3000
#
# Output:
# 1. CSV File: data/result.csv
#    Columns:
#      - l: Multipole moment (integer)
#      - TT: Raw temperature power spectrum C_l^TT (float, in muK^2)
# 2. Plot File: data/cmb_power_spectrum_1_<timestamp>.png
#    Description: Log-log plot of D_l^TT = l(l+1)C_l^TT/(2*pi) vs. multipole l.
#
# CAMB Version:
# The script will print the version of CAMB used during execution.
# ----------------------------------------------------------------------------------

import os
import camb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Ensure LaTeX rendering is off
plt.rcParams['text.usetex'] = False

def calculate_and_validate_cmb_power_spectrum():
    """
    Calculates the CMB temperature power spectrum using CAMB, saves the results,
    generates a plot, and validates the output CSV file.
    """
    # Print CAMB version
    try:
        print("Using CAMB version: " + str(camb.__version__))
    except AttributeError:
        print("Could not determine CAMB version. Please ensure CAMB is installed correctly.")

    
    # Create data directory if it doesn't exist
    database_path = "data"
    if not os.path.exists(database_path):
        os.makedirs(database_path)
        print("Created directory: " + str(database_path))

    # 1. Set up CAMB parameters
    pars = camb.CAMBparams()
    
    # Cosmological parameters
    h0_val = 74.0  # Hubble constant in km/s/Mpc
    ombh2_val = 0.022  # Baryon density * h^2
    omch2_val = 0.122  # Cold dark matter density * h^2
    mnu_val = 0.06  # Sum of neutrino masses in eV
    omk_val = 0.0  # Curvature Omega_k
    tau_val = 0.06  # Optical depth to reionization
    
    pars.set_cosmology(
        H0=h0_val, 
        ombh2=ombh2_val, 
        omch2=omch2_val, 
        mnu=mnu_val, 
        omk=omk_val, 
        tau=tau_val, 
        num_massive_neutrinos=1, 
        standard_neutrino_neff=3.046
    )
    
    # Primordial power spectrum parameters
    pars.InitPower.As = 2e-9  # Scalar amplitude
    pars.InitPower.ns = 0.965  # Scalar spectral index
    
    # Set maximum multipole and request lensed Cls
    lmax_val = 3000
    # lens_potential_accuracy=1 for good lensed Cls, higher for more accuracy
    pars.set_for_lmax(lmax_val, lens_potential_accuracy=1)
    
    # We want scalar Cls
    pars.WantScalars = True
    pars.WantTensors = False 
    pars.WantVectors = False

    # 2. Run CAMB
    print("Running CAMB to compute power spectra...")
    results = camb.get_results(pars)
    print("CAMB calculation complete.")

    # 3. Get lensed scalar Cls (C_l in muK^2 by default)
    # The output from get_lensed_scalar_cls is CL[0:lmax, TT, EE, BB, TE]
    # CAMB output Cls are already in muK^2 if CMB_unit='muK' (default)
    all_cls = results.get_lensed_scalar_cls(lmax=lmax_val)

    # 4. Extract l=2..3000 for TT
    # Multipole moments from l=2 to l=3000
    ls = np.arange(2, lmax_val + 1) # Integer array
    
    # TT spectrum is all_cls[:,0]. We need from l=2.
    # CAMB returns Cls from l=0. So, all_cls[l,0] is C_l^TT.
    cl_TT = all_cls[ls, 0] # Units: muK^2

    # 5. Save to CSV
    csv_filename = os.path.join(database_path, "result.csv")
    df_results = pd.DataFrame({'l': ls, 'TT': cl_TT})
    df_results.to_csv(csv_filename, index=False, float_format='%.8e')  # Save with scientific notation for precision
    print("Results saved to " + str(csv_filename))
    
    # Print head and tail of the dataframe for quick verification
    print("\nFirst 5 rows of the result.csv:")
    print(df_results.head().to_string())
    print("\nLast 5 rows of the result.csv:")
    print(df_results.tail().to_string())

    # 6. Validate CSV File
    print("\nValidating CSV file: " + str(csv_filename))
    try:
        df_loaded = pd.read_csv(csv_filename)
        validation_passed = True

        # Check columns
        expected_columns = ['l', 'TT']
        if list(df_loaded.columns) != expected_columns:
            print("Validation FAILED: Column names are incorrect. Expected: " + str(expected_columns) + ", Got: " + str(list(df_loaded.columns)))
            validation_passed = False
        else:
            print("Validation PASSED: Column names are correct.")

        # Check 'l' range and type
        if 'l' in df_loaded.columns:
            if not pd.api.types.is_integer_dtype(df_loaded['l']):
                print("Validation FAILED: 'l' column is not integer type.")
                validation_passed = False
            elif df_loaded['l'].min() != 2 or df_loaded['l'].max() != lmax_val:
                print("Validation FAILED: 'l' column range is incorrect. Expected min: 2, max: " + str(lmax_val) + 
                      ". Got min: " + str(df_loaded['l'].min()) + ", max: " + str(df_loaded['l'].max()))
                validation_passed = False
            elif len(df_loaded['l']) != (lmax_val - 2 + 1):
                print("Validation FAILED: Number of rows for 'l' is incorrect. Expected: " + str(lmax_val - 2 + 1) + 
                      ", Got: " + str(len(df_loaded['l'])))
                validation_passed = False
            else:
                print("Validation PASSED: 'l' column range and type are correct.")
        
        # Check 'TT' type and values
        if 'TT' in df_loaded.columns:
            if not pd.api.types.is_numeric_dtype(df_loaded['TT']):  # Checks for float or int
                print("Validation FAILED: 'TT' column is not numeric type.")
                validation_passed = False
            elif (df_loaded['TT'] <= 0).any():  # Power spectrum should be positive
                print("Validation FAILED: 'TT' column contains non-positive values.")
                print("Number of non-positive TT values: " + str((df_loaded['TT'] <= 0).sum()))
                validation_passed = False
            else:
                print("Validation PASSED: 'TT' column type is numeric and all values are positive.")
        
        if validation_passed:
            print("CSV file validation successful.")
        else:
            print("CSV file validation failed. Please check the messages above.")

    except FileNotFoundError:
        print("Validation FAILED: CSV file not found at " + str(csv_filename))
    except Exception as e:
        print("Validation FAILED: An error occurred during CSV validation: " + str(e))


    # 7. Create a visualization (plot)
    # Plot D_l^TT = l(l+1)C_l^TT / (2*pi)
    dl_TT = ls * (ls + 1) * cl_TT / (2 * np.pi)  # Units: muK^2

    plt.figure(figsize=(10, 6))
    plt.plot(ls, dl_TT, color='blue', linewidth=1.5)
    plt.title('CMB Temperature Power Spectrum ($D_l^{TT}$)')
    plt.xlabel('Multipole moment l')
    plt.ylabel('$l(l+1)C_l^{TT}/(2\pi)$ [$\mu K^2$]')  # Units: muK^2
    plt.grid(True, which="both", ls="--", linewidth=0.5)  # Grid for major and minor ticks
    plt.xscale('log') 
    plt.yscale('log') 
    
    # Add some specific tick marks for better readability on log scale
    xticks_major = [2, 10, 100, 1000, 3000]
    plt.xticks(xticks_major, [str(x) for x in xticks_major])
    
    # Ensure minor ticks are also shown if appropriate for the scale
    # Matplotlib usually handles minor ticks well with log scale, but can be customized if needed.

    plt.tight_layout()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(database_path, "cmb_power_spectrum_1_" + str(timestamp) + ".png")
    plt.savefig(plot_filename, dpi=300)
    print("\nPlot saved as: " + str(plot_filename))
    print("Description: Plot of the CMB temperature power spectrum D_l^TT = l(l+1)C_l^TT/(2*pi) vs multipole moment l (log-log scale).")
    print("The plot provides a visual check for the calculated power spectrum, showing acoustic peaks and damping tail.")
    plt.close()


if __name__ == '__main__':
    try:
        calculate_and_validate_cmb_power_spectrum()
        print("\nCMB power spectrum calculation, saving, and validation complete.")
    except ImportError:
        print("\nERROR: CAMB not found. Please ensure CAMB is installed.")
        # This script will just report the error.
        # The calling agent should handle installation if necessary.
    except Exception as e:
        print("\nAn error occurred during the script execution:")
        import traceback
        print(traceback.format_exc())
