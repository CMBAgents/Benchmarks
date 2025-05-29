# filename: codebase/calculate_cmb_power_spectrum.py
import os
import numpy as np
import pandas as pd
import camb

def calculate_cmb_power_spectrum():
    r"""
    Calculates the CMB temperature power spectrum for a specified non-flat Lambda CDM cosmology.

    The function uses the CAMB library to compute the temperature power spectrum
    D_l^TT = l(l+1)C_l^TT / (2*pi) in units of muK^2 for multipole moments
    l from 2 to 3000. The results are saved to a CSV file.

    Cosmological parameters used:
    - H0 (Hubble constant): 67.3 km/s/Mpc
    - ombh2 (Baryon density Omega_b * h^2): 0.022
    - omch2 (Cold dark matter density Omega_c * h^2): 0.122
    - mnu (Sum of neutrino masses): 0.06 eV
    - omk (Curvature Omega_k): 0.05
    - tau (Optical depth to reionization): 0.06
    - As (Scalar amplitude): 2e-9
    - ns (Scalar spectral index): 0.965
    - lmax (Maximum multipole moment): 3000

    The output is a CSV file named 'result.csv' saved in the 'data/' directory,
    containing two columns: 'l' and 'TT'.
    """
    print("Calculating CMB power spectrum with CAMB...")

    # Define cosmological parameters
    h0 = 67.3  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Baryon density
    omch2 = 0.122  # Cold dark matter density
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.05  # Curvature
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    lmax_calc = 3000  # Maximum multipole moment to calculate

    print("Cosmological parameters:")
    print("H0: " + str(h0) + " km/s/Mpc")
    print("ombh2: " + str(ombh2))
    print("omch2: " + str(omch2))
    print("mnu: " + str(mnu) + " eV")
    print("omk: " + str(omk))
    print("tau: " + str(tau))
    print("As: " + str(As))
    print("ns: " + str(ns))
    print("lmax: " + str(lmax_calc))
    
    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax=lmax_calc, lens_potential_accuracy=1)  # lens_potential_accuracy=1 for higher accuracy

    # Get results
    results = camb.get_results(pars)
    print("CAMB calculation complete.")

    # Get CMB power spectra
    # powers['total'] returns l(l+1)C_l/2pi in muK^2 for TT, EE, BB, TE
    # We need the TT part, which is the first column (index 0)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=False)
    cl_tt = powers['total'][:, 0]  # D_l^TT = l(l+1)C_l^TT / (2*pi)

    # We need l from 2 to lmax_calc (inclusive)
    # cl_tt array is indexed from l=0. So cl_tt[2] is for l=2.
    ls_for_df = np.arange(2, lmax_calc + 1)
    tt_for_df = cl_tt[2 : lmax_calc + 1]

    # Create pandas DataFrame
    df = pd.DataFrame({'l': ls_for_df, 'TT': tt_for_df})

    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Save to CSV
    file_path = 'data/result.csv'
    print("Saving results to " + file_path + "...")
    df.to_csv(file_path, index=False)
    
    print("CMB temperature power spectrum data saved to " + file_path)
    print("The results contain L values from " + str(ls_for_df[0]) + " to " + str(ls_for_df[-1]) + " and corresponding D_l^TT values.")
    print("First 5 rows of the result:")
    # Print dataframe head without truncation for better visibility of numbers
    print(df.head().to_string())


if __name__ == '__main__':
    calculate_cmb_power_spectrum()