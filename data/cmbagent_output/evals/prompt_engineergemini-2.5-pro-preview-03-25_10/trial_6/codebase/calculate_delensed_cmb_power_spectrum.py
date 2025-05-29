# filename: codebase/calculate_delensed_cmb_power_spectrum.py
import os
import numpy as np
import pandas as pd
import camb


def calculate_delensed_cmb_power_spectrum():
    r"""
    Calculates the delensed CMB temperature power spectrum for a flat Lambda CDM cosmology.

    The function uses CAMB to compute the lensed and unlensed scalar power spectra,
    then applies a delensing efficiency to obtain the delensed spectrum.
    The results, D_l = l(l+1)C_l^TT/(2pi) in muK^2, are saved to a CSV file.
    """

    # Cosmological Parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Physical baryon density
    omch2 = 0.122  # Physical cold dark matter density
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature parameter
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index

    # Calculation settings
    lmax_calc = 3050  # Max multipole for CAMB calculation (slightly higher for precision)
    lmax_out = 3000   # Max multipole for output
    delensing_efficiency = 0.80  # Delensing efficiency eta

    # Initialize CAMB parameters
    pars = camb.CAMBparams()

    # Set cosmological parameters
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)

    # Set initial power spectrum parameters
    # InitialPowerLaw specifies a power-law primordial spectrum P(k) = A_s (k/k_pivot)^(n_s-1)
    pars.set_initial_power(camb.initialpower.InitialPowerLaw(As=As, ns=ns))

    # Configure CAMB for lensed scalar spectra up to lmax_calc
    # lens_potential_accuracy=1 is default and generally good.
    # max_eta_k is important for lensed spectra accuracy at high l.
    pars.set_for_lmax(lmax_calc, max_eta_k=2 * lmax_calc, lens_potential_accuracy=1)
    
    # Ensure lensing is enabled (it's on by default if lens_potential_accuracy > 0)
    pars.DoLensing = True

    # Get results from CAMB
    results = camb.get_results(pars)

    # Get lensed and unlensed scalar power spectra
    # CMB_unit='muK' returns D_l = l(l+1)C_l/(2pi) in muK^2
    # spectra is a list of spectra to return: 'total', 'unlensed_scalar', 'lensed_scalar', 'tensor', 'lens_potential'
    power_spectra = results.get_cmb_power_spectra(lmax=lmax_out, spectra=['lensed_scalar', 'unlensed_scalar'], CMB_unit='muK')

    # Extract TT power spectra (column 0)
    # These are arrays D_l starting from l=0 up to lmax_out
    Dl_lensed_TT = power_spectra['lensed_scalar'][:, 0]  # D_l^TT (lensed) in muK^2
    Dl_unlensed_TT = power_spectra['unlensed_scalar'][:, 0]  # D_l^TT (unlensed) in muK^2

    # Calculate delensed power spectrum
    # D_l_delensed = D_l_unlensed + (1 - eta) * (D_l_lensed - D_l_unlensed)
    Dl_delensed_TT = Dl_unlensed_TT + (1.0 - delensing_efficiency) * (Dl_lensed_TT - Dl_unlensed_TT)

    # Prepare data for CSV output
    # Multipole moments from l=2 to lmax_out
    ls = np.arange(2, lmax_out + 1)

    # Select the corresponding power spectrum values
    # CAMB output arrays are indexed from l=0, so Dl_delensed_TT[l] corresponds to multipole l.
    Dl_delensed_TT_output = Dl_delensed_TT[2 : lmax_out + 1]

    # Create pandas DataFrame
    df_results = pd.DataFrame({
        'l': ls,
        'TT': Dl_delensed_TT_output  # Delensed D_l^TT in muK^2
    })

    # Create data directory if it doesn't exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Created directory: " + data_dir)

    # Save results to CSV
    file_path = os.path.join(data_dir, 'result.csv')
    df_results.to_csv(file_path, index=False)

    print("Delensed CMB temperature power spectrum calculation complete.")
    print("Results saved to: " + file_path)
    
    # Print some sample values from the DataFrame
    print("\nSample of the first 5 rows of the saved data:")
    print(df_results.head().to_string())
    print("\nSample of the last 5 rows of the saved data:")
    print(df_results.tail().to_string())
    print("\nDescription of the data:")
    # Set pandas display options for better output of describe()
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 10)
    print(df_results.describe().to_string())


if __name__ == '__main__':
    calculate_delensed_cmb_power_spectrum()