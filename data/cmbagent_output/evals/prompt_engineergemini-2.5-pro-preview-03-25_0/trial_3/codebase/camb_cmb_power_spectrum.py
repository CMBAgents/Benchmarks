# filename: codebase/camb_cmb_power_spectrum.py
import os
import numpy as np
import pandas as pd
import camb

def calculate_cmb_power_spectrum():
    r"""
    Calculates the CMB temperature power spectrum for a flat Lambda CDM cosmology
    using CAMB and saves the results to a CSV file.

    The cosmological parameters are:
    - Hubble constant (H0): 67.5 km/s/Mpc
    - Baryon density (ombh2): 0.02
    - Cold dark matter density (omch2): 0.122
    - Neutrino mass sum (mnu): 0.06 eV
    - Curvature (omk): 0 (flat cosmology)
    - Optical depth to reionization (tau): 0.06
    - Scalar amplitude (As): 2e-9
    - Scalar spectral index (ns): 0.965

    The function computes Dl_TT = l(l+1)C_l^TT/(2*pi) in muK^2
    for multipole moments l = 2 to l = 3000.
    The results are saved in 'data/result.csv'.
    """

    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Created directory: " + data_dir)
    else:
        print("Directory " + data_dir + " already exists.")

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.02, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965)
    
    lmax_calc = 3000
    # lens_potential_accuracy=1 enables lensing calculation, which is standard for lmax > ~1000
    pars.set_for_lmax(lmax_calc, lens_potential_accuracy=1)

    # Calculate results
    print("Running CAMB to get cosmological results...")
    results = camb.get_results(pars)
    print("CAMB calculation finished.")

    # Get total lensed CLs.
    # results.get_total_cls(lmax) returns D_l = l(l+1)C_l/(2pi) in muK^2.
    # all_cls is a numpy array, shape (nspectra, lmax+1), where nspectra is typically 4 (TT, EE, BB, TE).
    # all_cls[0,:] corresponds to D_l^TT.
    print("Extracting CMB power spectra...")
    all_cls = results.get_total_cls(lmax=lmax_calc) 

    # Extract D_l^TT for l from 0 to lmax_calc.
    # dl_tt is an array of D_l^TT values, indexed by l. So, dl_tt[l] is for multipole l.
    dl_tt = all_cls[0] # This is D_l^TT = l(l+1)C_l^TT/(2pi) in muK^2

    # We need l from 2 to 3000 as per requirements.
    l_min_output = 2
    l_max_output = 3000 
    
    # Generate array of l values for the output
    # These are integer multipole moments from l_min_output to l_max_output inclusive.
    l_values = np.arange(l_min_output, l_max_output + 1)
    
    # Extract the corresponding D_l^TT values.
    # dl_tt is 0-indexed by l, so dl_tt[l] is the power spectrum at multipole l.
    # We need values from dl_tt[l_min_output] up to dl_tt[l_max_output].
    dl_tt_values = dl_tt[l_min_output : l_max_output + 1]

    # Create pandas DataFrame
    df = pd.DataFrame({
        'l': l_values,      # Multipole moment l
        'TT': dl_tt_values  # D_l^TT = l(l+1)C_l^TT/(2pi) in muK^2
    })

    # Save to CSV
    file_path = os.path.join(data_dir, "result.csv")
    df.to_csv(file_path, index=False)

    print("\nCMB temperature power spectrum calculation complete.")
    print("Results saved to: " + str(file_path))
    print("\nCosmological parameters used:")
    print("  Hubble constant (H0): 67.5 km/s/Mpc")
    print("  Baryon density (ombh2): 0.02")
    print("  Cold dark matter density (omch2): 0.122")
    print("  Neutrino mass sum (mnu): 0.06 eV")
    print("  Curvature (omk): 0 (flat cosmology)")
    print("  Optical depth to reionization (tau): 0.06")
    print("  Scalar amplitude (As): 2e-9")
    print("  Scalar spectral index (ns): 0.965")
    
    print("\nOutput file 'result.csv' contains columns 'l' and 'TT'.")
    print("'TT' is l(l+1)C_l^TT/(2pi) in units of muK^2.")
    
    # Set pandas display options for better console output
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.width', 100) # Adjust width for better display

    print("\nShowing first 5 rows of the data:")
    print(df.head().to_string())
    print("\nShowing last 5 rows of the data:")
    print(df.tail().to_string())
    
    print("\nSpecific D_l^TT values (muK^2) from the saved data:")
    specific_ls_to_print = [2, 10, 100, 220, 500, 1000, 1500, 2000, 2500, 3000]
    for l_val in specific_ls_to_print:
        if l_val >= l_min_output and l_val <= l_max_output:
            # Retrieve from DataFrame to ensure we print what's saved
            tt_val_series = df.loc[df['l'] == l_val, 'TT']
            if not tt_val_series.empty:
                tt_val = tt_val_series.iloc[0]
                print("  l = " + str(l_val) + ": " + str(tt_val))
            else:
                # This should not happen if l_val is within the DataFrame's l range
                print("  l = " + str(l_val) + ": Not found in DataFrame (unexpected).")
        else:
            # This case handles l_val outside the saved range [2, 3000]
            # but potentially calculated by CAMB (e.g. l=0, 1 or l > 3000 if lmax_calc was larger)
            if l_val < dl_tt.shape[0] and l_val >= 0:
                 print("  l = " + str(l_val) + ": " + str(dl_tt[l_val]) + " (raw CAMB output, not in CSV range)")
            else:
                print("  l = " + str(l_val) + ": Not calculated (l < 0 or l > lmax_calc=" + str(lmax_calc) + ")")


if __name__ == '__main__':
    calculate_cmb_power_spectrum()
