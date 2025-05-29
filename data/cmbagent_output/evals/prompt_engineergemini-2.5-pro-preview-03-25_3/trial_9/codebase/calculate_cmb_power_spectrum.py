# filename: codebase/calculate_cmb_power_spectrum.py
r"""Calculates the CMB raw temperature power spectrum using CAMB and saves it to a CSV file."""
import os
import camb
import numpy as np
import pandas as pd

def calculate_cmb_power_spectrum():
    r"""Calculates the CMB raw temperature power spectrum for a flat Lambda CDM cosmology
    using specified parameters with CAMB. Saves the results to a CSV file."""
    # Cosmological parameters
    h0 = 74.0  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Baryon density Omega_b * h^2
    omch2 = 0.122  # Cold dark matter density Omega_c * h^2
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature Omega_k
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    lmax_calc = 3000  # Maximum multipole moment to calculate

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    
    # We want unlensed scalar Cls.
    # lens_potential_accuracy=0 means unlensed.
    # Max lmax for CMB set to lmax_calc.
    pars.set_for_lmax(lmax=lmax_calc, lens_potential_accuracy=0)
    
    # Calculate results
    results = camb.get_results(pars)

    # Get raw C_l values in muK^2
    # get_cmb_power_spectra returns a dictionary of spectra
    # 'unlensed_scalar': Cls for TT, EE, BB, TE (unlensed)
    # 'total': Cls for TT, EE, BB, TE (lensed)
    # We need unlensed_scalar for "raw" power spectrum.
    # raw_cl=True gives C_l, CMB_unit='muK' gives units of muK^2
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
    
    # Extract unlensed scalar C_l^TT
    # powers['unlensed_scalar'] is an array with shape (lmax_calc+1, 4)
    # Columns are TT, EE, BB, TE
    # We need TT, which is the first column (index 0)
    # CAMB output Cls are indexed from l=0.
    # We need l from 2 to lmax_calc.
    cl_tt_unlensed = powers['unlensed_scalar'][:, 0] # units: muK^2

    # Prepare data for CSV
    # Multipole moments 'l' from 2 to lmax_calc
    l_values = np.arange(2, lmax_calc + 1)
    
    # Corresponding C_l^TT values
    # cl_tt_unlensed[l] corresponds to C_l. So we need elements from index 2 to lmax_calc.
    tt_values = cl_tt_unlensed[2:lmax_calc + 1]

    # Create pandas DataFrame
    df = pd.DataFrame({'l': l_values, 'TT': tt_values})

    # Ensure 'data' directory exists
    if not os.path.exists("data"):
        os.makedirs("data")

    # Save to CSV
    file_path = "data/result.csv"
    df.to_csv(file_path, index=False)

    print("CMB temperature power spectrum calculation complete.")
    print("Results saved to " + file_path)
    print("Columns: l, TT (C_l^TT in muK^2)")
    print("Number of l values: " + str(len(df)))
    print("l range: " + str(df['l'].min()) + " to " + str(df['l'].max()))
    print("\nFirst 5 rows of the data:")
    print(df.head().to_string()) 
    print("\nLast 5 rows of the data:")
    print(df.tail().to_string())

# Direct execution without try-except for ImportError
calculate_cmb_power_spectrum()
