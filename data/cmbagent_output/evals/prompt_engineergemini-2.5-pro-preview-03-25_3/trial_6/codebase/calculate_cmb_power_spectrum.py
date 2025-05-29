# filename: codebase/calculate_cmb_power_spectrum.py
import camb
from camb import model, initialpower
import numpy as np
import pandas as pd
import os


def calculate_cmb_power_spectrum():
    r"""
    Calculates the CMB raw temperature power spectrum (C_l^TT) for a flat Lambda CDM cosmology.

    The function uses specified cosmological parameters, computes the spectrum using CAMB,
    converts it to units of muK^2, and saves it to a CSV file.
    It also prints a summary of parameters and data.
    """
    # Cosmological parameters
    H0_val = 74.0  # Hubble constant in km/s/Mpc
    ombh2_val = 0.022  # Baryon density Omega_b * h^2
    omch2_val = 0.122  # Cold dark matter density Omega_c * h^2
    mnu_val = 0.06  # Sum of neutrino masses in eV
    omk_val = 0.0  # Curvature Omega_k
    tau_val = 0.06  # Optical depth to reionization
    As_val = 2e-9  # Scalar amplitude
    ns_val = 0.965  # Scalar spectral index
    lmax_calc = 3000 # Maximum multipole moment

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0_val, ombh2=ombh2_val, omch2=omch2_val, mnu=mnu_val, omk=omk_val, tau=tau_val)
    pars.InitPower.set_params(As=As_val, ns=ns_val)

    # Configure for unlensed scalar Cls up to lmax
    # lens_potential_accuracy=0 means unlensed.
    pars.set_for_lmax(lmax_calc, lens_potential_accuracy=0)
    
    # We want scalar TT power spectrum, ensure unlensed
    pars.WantScalars = True
    pars.WantTensors = False # Default
    pars.WantVectors = False # Default
    pars.DoLensing = False # Ensure unlensed Cls are calculated by get_unlensed_scalar_cls

    # Get results
    print("Running CAMB to calculate power spectra...")
    results = camb.get_results(pars)
    print("CAMB calculation finished.")

    # Get unlensed scalar Cls
    # This returns a dictionary of Cls for TT, EE, BB, TE.
    # Values are C_l in K^2 units. Arrays are indexed from l=0.
    powers = results.get_unlensed_scalar_cls(lmax=lmax_calc)

    # Extract TT power spectrum (C_l^TT)
    cl_TT_K2 = powers['T']  # C_l^TT in K^2

    # We need l from 2 to lmax_calc (inclusive)
    ls = np.arange(2, lmax_calc + 1) # Multipole moments from 2 to 3000

    # Extract the relevant part of cl_TT_K2 for l >= 2
    # cl_TT_K2[l] corresponds to multipole l. So cl_TT_K2[2] is for l=2.
    cl_TT_K2_slice = cl_TT_K2[2:lmax_calc + 1]

    # Convert C_l^TT from K^2 to muK^2
    # 1 K = 10^6 muK, so 1 K^2 = (10^6 muK)^2 = 10^12 muK^2
    conversion_factor_K2_to_muK2 = (1e6)**2
    cl_TT_muK2 = cl_TT_K2_slice * conversion_factor_K2_to_muK2 # C_l^TT in muK^2

    # Create pandas DataFrame
    df = pd.DataFrame({'l': ls, 'TT': cl_TT_muK2})

    # Ensure 'l' column is integer
    df['l'] = df['l'].astype(int)

    # Save to CSV
    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: " + output_dir)
    
    csv_filepath = os.path.join(output_dir, 'result.csv')
    df.to_csv(csv_filepath, index=False, float_format='%.8e') # Use scientific notation for precision
    
    # Print summary
    print("\nCMB Temperature Power Spectrum Calculation Complete.")
    print("Parameters used:")
    print("H0: " + str(pars.H0) + " km/s/Mpc")
    print("ombh2 (Omega_b h^2): " + str(pars.ombh2))
    print("omch2 (Omega_c h^2): " + str(pars.omch2))
    print("mnu (Sum of neutrino masses): " + str(pars.Mnu) + " eV")
    print("omk (Curvature Omega_k): " + str(pars.omk))
    # CAMB stores tau in pars.Reion.optical_depth after set_cosmology
    print("tau (Optical depth to reionization): " + str(pars.Reion.optical_depth))
    print("As (Scalar amplitude): " + str(pars.InitPower.As))
    print("ns (Scalar spectral index): " + str(pars.InitPower.ns))
    
    print("\nOutput format: C_l^TT in muK^2")
    print("Saved to: " + csv_filepath)
    
    print("\nFirst 5 rows of the data (l, TT):")
    # Configure pandas to display floats with more precision for print
    with pd.option_context('display.float_format', '{:.6e}'.format):
        print(df.head())
    
    print("\nLast 5 rows of the data (l, TT):")
    with pd.option_context('display.float_format', '{:.6e}'.format):
        print(df.tail())


if __name__ == '__main__':
    calculate_cmb_power_spectrum()