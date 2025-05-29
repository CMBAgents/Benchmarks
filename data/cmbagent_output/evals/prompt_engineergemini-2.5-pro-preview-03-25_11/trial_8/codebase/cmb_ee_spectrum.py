# filename: codebase/cmb_ee_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def calculate_cmb_ee_spectrum():
    r"""
    Calculates the CMB E-mode polarization power spectrum D_l^EE = l(l+1)C_l^EE/(2pi)
    for a specific flat Lambda CDM cosmology using CAMB.

    The cosmological parameters are:
    - H0: 67.5 km/s/Mpc
    - Omega_b h^2: 0.022
    - Omega_c h^2: 0.122
    - A_s: 1.8e-9 * exp(2 * tau)
    - n_s: 0.95
    - tau: 0.1

    Reionization model: Tanh model with ReionizationExponent = 2.0 and symmetric_reionization = False,
    calibrated to the given tau.

    The E-mode power spectrum is computed in muK^2 for l from 2 to 100.
    Results are saved to 'data/result.csv' and printed to the console.
    """

    # Cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Baryon density
    omch2 = 0.122  # Cold dark matter density
    tau = 0.1  # Optical depth to reionization
    
    # Scalar amplitude A_s, as defined in the problem
    # A_s is the primordial scalar amplitude at k_pivot = 0.05 Mpc^-1
    As_val = 1.8e-9 * np.exp(2 * tau) 
    
    ns = 0.95  # Scalar spectral index
    
    lmax_calc = 100  # Maximum multipole moment to calculate

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    
    # Set cosmology
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0, tau=tau)
    # mnu=0.06 eV is CAMB's default for one massive neutrino species.
    # omk=0 for a flat universe.
    # tau=tau sets Reion.optical_depth and Reion.use_optical_depth = True

    # Set initial power spectrum parameters
    pars.InitPower.As = As_val
    pars.InitPower.ns = ns

    # Set reionization model details
    # The prompt specifies "Exponential reionization with exponent power 2".
    # This is interpreted as CAMB's Tanh model with ReionizationExponent = 2.0.
    # To use ReionizationExponent, symmetric_reionization must be False.
    pars.Reion.ReionizationModel = 'Tanh'
    pars.Reion.symmetric_reionization = False
    pars.Reion.ReionizationExponent = 2.0
    # CAMB will adjust Reion.ReionizationRedshift to match the specified tau
    # using this Tanh model configuration.

    # Set calculation accuracy and requested spectra
    # We need unlensed scalar EE spectrum.
    pars.set_for_lmax(lmax_calc, lens_potential_accuracy=0)  # No lensing
    pars.WantTensors = False  # Only scalar modes

    # Get results
    results = camb.get_results(pars)

    # Get CMB power spectra D_l = l(l+1)C_l/(2pi)
    # CMB_unit='muK' ensures D_l is in muK^2
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax_calc)
    
    # EE spectrum is in the second column (index 1) of 'unlensed_scalar'
    # powers['unlensed_scalar'] has shape (lmax_calc+1, 4) for TT, EE, BB, TE
    # It is 0-indexed by l, so powers['unlensed_scalar'][l] is for multipole l.
    unlensed_scalar_cls = powers['unlensed_scalar']
    
    # We need l from 2 to 100
    l_values = np.arange(2, lmax_calc + 1)
    
    # Extract D_l^EE for the required l range
    # D_l^EE values are unlensed_scalar_cls[l, 1]
    EE_values = unlensed_scalar_cls[l_values, 1]  # Unit: muK^2

    # Create pandas DataFrame
    df_results = pd.DataFrame({'l': l_values, 'EE': EE_values})

    # Ensure 'l' column is integer
    df_results['l'] = df_results['l'].astype(int)

    # Create data directory if it doesn't exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save results to CSV
    csv_filename = os.path.join(data_dir, 'result.csv')
    df_results.to_csv(csv_filename, index=False)

    print("CMB E-mode polarization power spectrum calculation complete.")
    print("Results saved to: " + str(csv_filename))
    
    # Print the full DataFrame to console as requested
    print("\nCalculated E-mode power spectrum (l(l+1)C_l^EE / (2pi) in muK^2):")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(df_results)

if __name__ == '__main__':
    calculate_cmb_ee_spectrum()