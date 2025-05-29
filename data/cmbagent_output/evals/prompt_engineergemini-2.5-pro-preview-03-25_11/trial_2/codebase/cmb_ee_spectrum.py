# filename: codebase/cmb_ee_spectrum.py
import numpy as np
import pandas as pd
import camb
import os

def calculate_cmb_ee_spectrum():
    r"""
    Calculates the CMB E-mode polarization power spectrum D_l^EE = l(l+1)C_l^EE/(2pi)
    for a flat Lambda CDM cosmology using specified parameters with CAMB.

    The D_l^EE values are in muK^2.
    Results are saved to 'data/result.csv'.
    """

    # Cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Baryon density Omega_b * h^2
    omch2 = 0.122  # Cold dark matter density Omega_c * h^2
    tau = 0.1  # Optical depth to reionization
    
    # Scalar amplitude As = (A_s_factor) * exp(2*tau)
    # where A_s_factor = 1.8e-9 is interpreted as A_s*exp(-2*tau)
    As_val = 1.8e-9 * np.exp(2 * tau)  # Scalar amplitude A_s
    
    ns = 0.95  # Scalar spectral index n_s
    lmax_calc = 100  # Maximum multipole moment to calculate up to

    # Create CAMBparams object
    pars = camb.CAMBparams()

    # Set cosmology
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0, tau=tau)

    # Set initial power spectrum parameters
    pars.InitPower.As = As_val
    pars.InitPower.ns = ns

    # Set reionization model details
    # The prompt "Exponential reionization with exponent power 2" is interpreted as
    # CAMB's Tanh reionization model with ReionizationExponent = 2.0.
    # CAMB will adjust ReionizationRedshift to match the specified tau.
    pars.Reion.Reionization = True  # Ensure reionization is on
    pars.Reion.ReionizationModel = 'Tanh'  # Explicitly set, though default if Reion.Reionization is True
    pars.Reion.ReionizationExponent = 2.0  # Set the exponent for the Tanh model

    # Set calculations for lmax and request unlensed scalar spectra
    pars.set_for_lmax(lmax_calc, lens_potential_accuracy=0)  # lens_potential_accuracy=0 for unlensed
    pars.WantTensors = False  # We only want scalar modes

    # Get results
    results = camb.get_results(pars)

    # Get CMB power spectra D_l = l(l+1)C_l/(2pi)
    # CMB_unit='muK' gives D_l in muK^2
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax_calc)

    # Extract unlensed scalar EE spectrum
    # powers['unlensed_scalar'] is an array with columns: TT, EE, BB, TE
    # It is indexed by l, from l=0 to lmax_calc.
    # D_l^EE is in the second column (index 1).
    EE_spectrum_all_ls = powers['unlensed_scalar'][:, 1]  # D_l^EE in muK^2

    # We need l from 2 to 100
    ls = np.arange(2, lmax_calc + 1)
    EE_values = EE_spectrum_all_ls[ls]

    # Create pandas DataFrame
    df_results = pd.DataFrame({'l': ls, 'EE': EE_values})

    # Create data directory if it doesn't exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save to CSV
    file_path = os.path.join(data_dir, 'result.csv')
    df_results.to_csv(file_path, index=False)

    print("CMB E-mode power spectrum calculation complete.")
    print("First 5 rows of the E-mode power spectrum data (D_l^EE in muK^2):")
    # Print with higher precision for EE values
    with pd.option_context('display.float_format', '{:.6e}'.format):
        print(df_results.head())
    print("\nFull E-mode power spectrum data saved to " + file_path)


if __name__ == '__main__':
    calculate_cmb_ee_spectrum()
