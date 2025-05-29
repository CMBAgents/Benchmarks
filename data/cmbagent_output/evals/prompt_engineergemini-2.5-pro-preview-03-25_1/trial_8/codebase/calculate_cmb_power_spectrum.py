# filename: codebase/calculate_cmb_power_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def calculate_cmb_power_spectrum():
    r"""
    Calculates the CMB temperature power spectrum for a non-flat Lambda CDM cosmology.

    The function uses specified cosmological parameters to compute the
    temperature power spectrum D_l^TT = l(l+1)C_l^TT/(2pi) in units of muK^2
    for multipole moments l from 2 to 3000. The results are saved in a
    CSV file named 'result.csv' in the 'data/' directory.
    """

    # Cosmological Parameters
    H0 = 67.3  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Baryon density parameter
    omch2 = 0.122  # Cold dark matter density parameter
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.05  # Curvature density parameter
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    lmax = 3000 # Maximum multipole moment

    # Print parameters to console
    print("Calculating CMB power spectrum with the following parameters:")
    print("H0: " + str(H0) + " km/s/Mpc")
    print("ombh2: " + str(ombh2))
    print("omch2: " + str(omch2))
    print("mnu: " + str(mnu) + " eV")
    print("omk: " + str(omk))
    print("tau: " + str(tau))
    print("As: " + str(As))
    print("ns: " + str(ns))
    print("lmax: " + str(lmax))
    print("\n")

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=1) # lens_potential_accuracy=1 for standard accuracy

    # Get results
    results = camb.get_results(pars)

    # Get total (lensed scalar) CMB power spectra
    # powers is an array with columns TT, EE, BB, TE
    # D_l = l(l+1)C_l/(2pi) in muK^2
    powers = results.get_total_cls(lmax=lmax) # Powers is (lmax+1) x N_spectra (usually 4 for TT, EE, BB, TE)

    # Extract D_l^TT for l from 2 to lmax
    # powers[l,0] is D_l^TT
    # We need l from 2 to 3000
    ls = np.arange(2, lmax + 1)
    dl_TT = powers[2:lmax+1, 0]  # D_l^TT in muK^2

    # Create DataFrame
    df_results = pd.DataFrame({'l': ls, 'TT': dl_TT})

    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save results to CSV
    file_path = os.path.join(data_dir, "result.csv")
    df_results.to_csv(file_path, index=False)

    print("CMB temperature power spectrum calculation complete.")
    print("Results saved to: " + file_path)
    print("\nFirst 5 rows of the results:")
    # Print head of the dataframe, ensuring it's not truncated
    with pd.option_context('display.max_rows', None, 
                           'display.max_columns', None, 
                           'display.width', 1000):
        print(df_results.head())

if __name__ == '__main__':
    calculate_cmb_power_spectrum()