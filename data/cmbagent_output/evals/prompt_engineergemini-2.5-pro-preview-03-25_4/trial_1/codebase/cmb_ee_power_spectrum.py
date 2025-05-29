# filename: codebase/cmb_ee_power_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def calculate_cmb_ee_spectrum():
    r"""
    Calculates the CMB E-mode polarization power spectrum l(l+1)C_l^{EE}/(2pi)
    for a flat Lambda CDM cosmology using specified parameters with CAMB.

    The function computes the E-mode power spectrum in units of muK^2
    for multipole moments from l=2 to l=3000 and saves the results
    in a CSV file.

    Cosmological Parameters:
        Hubble constant (H0): 67.5 km/s/Mpc
        Baryon density (ombh2): 0.022
        Cold dark matter density (omch2): 0.122
        Neutrino mass sum (mnu): 0.06 eV
        Curvature (omk): 0
        Optical depth to reionization (tau): 0.04
        Scalar amplitude (As): 2e-9
        Scalar spectral index (ns): 0.965
    """
    # Set up CAMB parameters
    pars = camb.CAMBparams()

    # Set cosmological parameters
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.04)
    pars.InitPower.set_params(As=2e-9, ns=0.965)

    # Set lmax for calculation
    lmax_calc = 3000
    pars.set_for_lmax(lmax_calc, lens_potential_accuracy=1)  # lens_potential_accuracy=1 is default

    # Calculate results
    results = camb.get_results(pars)

    # Get CMB power spectra.
    # get_cmb_power_spectra returns l(l+1)C_l/2pi by default (raw_cl=False)
    # CMB_unit='muK' ensures output is in muK^2
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=False)
    
    # The 'total' key contains lensed scalar Cls.
    # Columns are: TT, EE, BB, TE
    # We need the EE spectrum, which is the second column (index 1).
    # The array is indexed by l, starting from l=0.
    # For l=0 and l=1, EE power is typically zero or not well-defined.
    
    # Extract l values and EE spectrum
    # The output array `powers['total']` has shape (lmax_calc+1, 4)
    # ls go from 0 to lmax_calc
    ls = np.arange(powers['total'].shape[0])
    cl_ee = powers['total'][:, 1]  # EE is the second column

    # Create a DataFrame
    df_data = {'l': ls, 'EE': cl_ee}
    df = pd.DataFrame(df_data)

    # Filter for l from 2 to 3000
    df_filtered = df[(df['l'] >= 2) & (df['l'] <= 3000)].copy()
    df_filtered['l'] = df_filtered['l'].astype(int)

    # Define the data directory
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save to CSV
    file_path = os.path.join(data_dir, "result.csv")
    df_filtered.to_csv(file_path, index=False)

    print("CMB E-mode polarization power spectrum calculation complete.")
    print("Results saved to: " + str(file_path))
    print("First 5 rows of the E-mode power spectrum data (l(l+1)C_l^{EE}/(2pi) in muK^2):")
    # Configure pandas to display more rows/columns if needed for verification
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 5)
    pd.set_option('display.width', 100)
    print(df_filtered.head())
    
    # Print some specific values for verification
    if len(df_filtered) > 0:
        print("\nSample values:")
        specific_ls = [2, 10, 100, 500, 1000, 2000, 3000]
        for l_val in specific_ls:
            if l_val in df_filtered['l'].values:
                ee_val = df_filtered[df_filtered['l'] == l_val]['EE'].iloc[0]
                print("l = " + str(l_val) + ", EE = " + str(ee_val) + " muK^2")
            elif l_val <= lmax_calc:  # if l_val is within requested range but not in df
                print("l = " + str(l_val) + " not found in the output range (check l_min).")


if __name__ == '__main__':
    calculate_cmb_ee_spectrum()