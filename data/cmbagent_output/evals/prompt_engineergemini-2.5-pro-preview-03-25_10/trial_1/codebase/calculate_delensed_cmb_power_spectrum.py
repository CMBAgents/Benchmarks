# filename: codebase/calculate_delensed_cmb_power_spectrum.py
import camb
import numpy as np
import pandas as pd
import os


def calculate_delensed_cmb_power_spectrum():
    r"""
    Calculates the delensed CMB temperature power spectrum D_l^TT = l(l+1)C_l^TT/(2pi)
    for a flat Lambda CDM cosmology using CAMB.

    The calculation uses specified cosmological parameters and a delensing efficiency of 80%.
    The results (l and D_l^TT in muK^2) for l=2 to 3000 are saved to 'data/result.csv'.
    """

    # Cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Baryon density
    omch2 = 0.122  # Cold dark matter density
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0  # Curvature
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    lmax = 3000  # Maximum multipole moment
    delensing_efficiency = 0.8

    print("Calculating delensed CMB power spectrum with CAMB.")
    print("Cosmological Parameters:")
    print("H0: " + str(H0) + " km/s/Mpc")
    print("ombh2: " + str(ombh2))
    print("omch2: " + str(omch2))
    print("mnu: " + str(mnu) + " eV")
    print("omk: " + str(omk))
    print("tau: " + str(tau))
    print("As: " + str(As))
    print("ns: " + str(ns))
    print("lmax: " + str(lmax))
    print("Delensing efficiency: " + str(delensing_efficiency))
    print("\n")

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=1)  # lens_potential_accuracy=1 for standard
    pars.DoLensing = True  # Ensure lensed C_ls are computed, unlensed are also available

    # Get results from CAMB
    results = camb.get_results(pars)

    # Get power spectra D_l = l(l+1)C_l/(2pi) in muK^2
    # 'unlensed_scalar': unlensed scalar Cls (TT, EE, BB, TE)
    # 'lensed_scalar': lensed scalar Cls (TT, EE, BB, TE)
    # These are D_l's in muK^2 because CMB_unit='muK' and raw_cl=False are defaults
    # for these D_l quantities when obtained via get_cmb_power_spectra.
    # To be explicit, we pass these arguments.
    power_spectra = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=False)

    # Extract TT power spectra (first column, index 0)
    # These arrays are indexed from l=0 up to lmax.
    dl_unlensed_tt_muK2 = power_spectra['unlensed_scalar'][:, 0]
    dl_lensed_tt_muK2 = power_spectra['lensed_scalar'][:, 0]

    # Calculate delensed TT power spectrum
    # D_l_delensed = D_l_unlensed + (1 - efficiency) * (D_l_lensed - D_l_unlensed)
    dl_delensed_tt_muK2 = dl_unlensed_tt_muK2 + (1 - delensing_efficiency) * (dl_lensed_tt_muK2 - dl_unlensed_tt_muK2)

    # Prepare data for CSV file (l from 2 to lmax)
    ls = np.arange(2, lmax + 1)
    dl_delensed_tt_muK2_selected = dl_delensed_tt_muK2[2 : lmax + 1]

    # Create DataFrame
    df_results = pd.DataFrame({
        'l': ls,
        'TT': dl_delensed_tt_muK2_selected  # D_l^TT in muK^2
    })

    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Save to CSV
    file_path = os.path.join(data_dir, "result.csv")
    df_results.to_csv(file_path, index=False)

    print("Delensed CMB temperature power spectrum D_l^TT = l(l+1)C_l^TT/(2pi) [muK^2]")
    print("Results saved to " + file_path)
    
    # Print summary of the DataFrame
    pd.set_option('display.max_rows', 10)  # For concise printing
    pd.set_option('display.width', 1000)  # Adjust width for better display
    print("\nFirst 5 rows of the data:")
    print(df_results.head(5).to_string())
    print("\nLast 5 rows of the data:")
    print(df_results.tail(5).to_string())


if __name__ == '__main__':
    calculate_delensed_cmb_power_spectrum()