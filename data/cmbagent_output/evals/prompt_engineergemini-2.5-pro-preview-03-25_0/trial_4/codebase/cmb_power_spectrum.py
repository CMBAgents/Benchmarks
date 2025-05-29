# filename: codebase/cmb_power_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def calculate_cmb_power_spectrum():
    r"""
    Calculates the CMB temperature power spectrum for a flat Lambda CDM cosmology
    using specified parameters with CAMB.

    The script computes the temperature power spectrum (l(l+1)C_l^TT / (2pi))
    in units of muK^2 for multipole moments from l=2 to l=3000.
    The results are saved in a CSV file named 'result.csv' in the 'data/' directory.
    """

    # Cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.02  # Baryon density * h^2
    omch2 = 0.122  # Cold dark matter density * h^2
    mnu = 0.06  # Neutrino mass sum in eV
    omk = 0.0  # Curvature (0 for flat)
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    lmax_calc = 3000 # Max multipole moment to calculate for

    # Create CAMBparams object
    pars = camb.CAMBparams()

    # Set cosmological parameters
    # H0: Hubble constant at z=0 (km/s/Mpc)
    # ombh2: Physical baryon density parameter
    # omch2: Physical cold dark matter density parameter
    # mnu: Sum of neutrino masses (eV)
    # omk: Omega_k, curvature density parameter
    # tau: Optical depth to reionization
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)

    # Set initial power spectrum parameters
    # As: Scalar fluctuation amplitude (at k=0.05 Mpc^-1)
    # ns: Scalar spectral index
    pars.InitPower.set_params(As=As, ns=ns)

    # Set calculation options
    # lmax: Maximum multipole l to calculate.
    pars.set_for_lmax(lmax=lmax_calc, max_eta_k=None, k_eta_max_scalar=None)
    
    # We want scalar Cls.
    pars.WantScalars = True
    # We want lensed CMB Cls for accuracy, as is standard.
    pars.DoLensing = True 
    
    # Get results from CAMB
    results = camb.get_results(pars)

    # Get CMB power spectra
    # CMB_unit='muK' ensures Cls are in muK^2.
    # raw_cl=False (default) means it returns l(l+1)Cl/(2pi).
    # This provides D_l = l(l+1)C_l/(2pi)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=False)

    # The 'total' array contains columns: l, D_TT, D_EE, D_BB, D_TE
    # D_X = l(l+1)C_l^X / (2pi)
    # We need D_TT, which is the second column (index 1).
    all_cls = powers['total']
    # ls_camb: array of multipole moments (L)
    ls_camb = all_cls[:, 0]
    # dl_tt_camb: array of D_L^TT = L(L+1)C_L^TT / (2pi) in muK^2
    dl_tt_camb = all_cls[:, 1]

    # Filter for l from 2 to lmax_calc (inclusive)
    # CAMB output for C_l starts from l=0.
    # The problem asks for l from 2 to 3000.
    mask = (ls_camb >= 2) & (ls_camb <= lmax_calc)
    l_values = ls_camb[mask].astype(int) # Multipole moment l
    tt_spectrum = dl_tt_camb[mask] # D_l^TT in muK^2

    # Create DataFrame
    df_results = pd.DataFrame({'l': l_values, 'TT': tt_spectrum})

    # Define data directory
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Created directory: " + str(data_dir))

    # Save to CSV
    csv_filename = os.path.join(data_dir, "result.csv")
    df_results.to_csv(csv_filename, index=False)

    print("CMB temperature power spectrum calculation complete.")
    print("Results saved to: " + str(csv_filename))
    
    # Print detailed information about the DataFrame
    print("\nFirst 5 rows of the data:")
    print(df_results.head().to_string())
    print("\nLast 5 rows of the data:")
    print(df_results.tail().to_string())
    print("\nShape of the data (rows, columns):")
    print(df_results.shape)
    print("\nSummary statistics of the data:")
    print(df_results.describe().to_string())

    # Verification of the number of multipoles
    expected_rows = lmax_calc - 2 + 1
    if len(df_results) == expected_rows:
        print("\nNumber of multipole moments is correct: " + str(len(df_results)))
    else:
        print("\nWarning: Number of multipole moments is " + str(len(df_results)) + " , expected " + str(expected_rows))
        if not df_results.empty:
            print("Min l in output: " + str(l_values.min()))
            print("Max l in output: " + str(l_values.max()))
        else:
            print("Output DataFrame is empty.")


if __name__ == '__main__':
    calculate_cmb_power_spectrum()