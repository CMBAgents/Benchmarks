# filename: codebase/delensed_cmb_power_spectrum.py
r"""
Calculates the delensed CMB temperature power spectrum using CAMB.

This script sets up a flat Lambda CDM cosmology with specified parameters,
computes the lensed and unlensed scalar CMB power spectra using CAMB,
applies an 80% delensing efficiency to the temperature (TT) power spectrum,
and saves the results for multipoles l=2 to l=3000 into a CSV file.
The output power spectrum is l(l+1)C_l^{TT}/(2pi) in units of muK^2.
"""

import os
import numpy as np
import pandas as pd
import camb

def calculate_delensed_cmb_power_spectrum():
    r"""
    Calculates and saves the delensed CMB temperature power spectrum.
    
    The function performs the following steps:
    1. Defines cosmological parameters.
    2. Initializes and configures CAMB.
    3. Computes lensed and unlensed power spectra.
    4. Applies delensing efficiency.
    5. Saves the resulting delensed TT power spectrum to a CSV file.
    6. Prints a summary of the output.
    """
    # Ensure data directory exists
    database_path = 'data'
    os.makedirs(database_path, exist_ok=True)

    # Cosmological Parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Baryon density
    omch2 = 0.122  # Cold dark matter density
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0  # Curvature
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    lmax_calc = 3000  # Maximum multipole for calculation

    # CAMB Parameter Setup
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.WantCls = True
    pars.DoLensing = True 
    pars.set_for_lmax(lmax_calc, lens_potential_accuracy=1) # lens_potential_accuracy=1 is default

    # Get CAMB Results
    results = camb.get_results(pars)

    # Get Power Spectra (D_l = l(l+1)C_l/(2pi) in muK^2)
    # CAMB returns spectra from l=0 to lmax_calc
    # lensed_cls[0] is TT, lensed_cls[1] is EE, lensed_cls[2] is BB, lensed_cls[3] is TE
    lensed_cls = results.get_lensed_scalar_cls(lmax=lmax_calc) # Output is D_l in muK^2
    unlensed_cls = results.get_unlensed_scalar_cls(lmax=lmax_calc) # Output is D_l in muK^2
    
    # Extract TT components
    Dl_TT_lensed = lensed_cls[0, :]    # Units: muK^2
    Dl_TT_unlensed = unlensed_cls[0, :]  # Units: muK^2

    # Apply Delensing Efficiency
    delensing_efficiency = 0.80
    # Formula: D_l_delensed = (1 - efficiency) * D_l_lensed + efficiency * D_l_unlensed
    Dl_TT_delensed = (1 - delensing_efficiency) * Dl_TT_lensed + delensing_efficiency * Dl_TT_unlensed # Units: muK^2

    # Prepare data for CSV (l from 2 to 3000)
    # CAMB arrays are indexed from l=0. We need l=2 up to lmax_calc.
    ls = np.arange(2, lmax_calc + 1) # Multipole moments
    TT_values = Dl_TT_delensed[2 : lmax_calc + 1] # Corresponding D_l values in muK^2

    # Create Pandas DataFrame
    df_results = pd.DataFrame({'l': ls, 'TT': TT_values})

    # Save to CSV
    output_filename = os.path.join(database_path, 'result.csv')
    df_results.to_csv(output_filename, index=False)

    # Print summary
    print("Delensed CMB temperature power spectrum calculation complete.")
    print("Results saved to: " + output_filename)
    print("Columns: 'l' (multipole moment), 'TT' (l(l+1)C_l^TT/(2pi) in muK^2)")
    print("First 5 rows of the data:")
    # Using to_string() to ensure full display without truncation and avoid f-strings/format
    print(df_results.head().to_string())


if __name__ == '__main__':
    calculate_delensed_cmb_power_spectrum()
