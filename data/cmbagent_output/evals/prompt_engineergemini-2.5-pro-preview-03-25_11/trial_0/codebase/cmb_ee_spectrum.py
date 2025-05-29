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
    - Hubble constant (H0): 67.5 km/s/Mpc
    - Baryon density (ombh2): 0.022
    - Cold dark matter density (omch2): 0.122
    - Scalar amplitude (As): 1.8e-9 * exp(2 * tau)
    - Scalar spectral index (ns): 0.95
    - Optical depth to reionization (tau): 0.1
    - Reionization model: Exponential reionization with exponent power 2

    The E-mode power spectrum is computed in units of muK^2 for multipole
    moments l from 2 to 100.

    The results are saved in a CSV file 'data/result.csv' and printed to the console.
    """

    # Cosmological parameters
    H0_val = 67.5  # H0 in km/s/Mpc
    ombh2_val = 0.022  # Omega_b * h^2
    omch2_val = 0.122  # Omega_c * h^2
    tau_val = 0.1  # Optical depth to reionization
    As_val = 1.8e-9 * np.exp(2 * tau_val)  # Scalar amplitude A_s
    ns_val = 0.95  # Scalar spectral index n_s
    lmax_val = 100  # Maximum multipole moment

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0_val, ombh2=ombh2_val, omch2=omch2_val, tau=tau_val)
    pars.InitPower.As = As_val
    pars.InitPower.ns = ns_val

    # Set reionization model
    # pars.Reion.optical_depth is already set by set_cosmology(tau=tau_val)
    # CAMB will use this tau to calibrate the reionization model chosen.
    pars.Reion.model_str = 'ReionizationExp'  # Exponential reionization model
    pars.Reion.ReionizationExpPower = 2.0  # Exponent for the reionization x_e(z) profile

    # Set calculation range and accuracy (unlensed spectra)
    pars.set_for_lmax(lmax=lmax_val, lens_potential_accuracy=0)

    # Calculate results
    results = camb.get_results(pars)

    # Get CMB power spectra
    # CMB_unit='muK' gives C_l in muK^2
    # raw_cl=False (default) gives l(l+1)C_l/(2pi)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=False)

    # Extract unlensed scalar E-mode power spectrum D_l^EE
    # powers['unlensed_scalar'] has shape (4, lmax+1): TT, EE, BB, TE
    # Index 1 corresponds to EE.
    # Values are D_l = l(l+1)C_l/(2pi)
    dl_ee_all = powers['unlensed_scalar'][1, :]  # D_l^EE in muK^2

    # We need l from 2 to 100
    ls = np.arange(2, lmax_val + 1)
    dl_ee_values = dl_ee_all[2 : lmax_val + 1]

    # Create pandas DataFrame
    df_results = pd.DataFrame({'l': ls, 'EE': dl_ee_values})
    df_results['l'] = df_results['l'].astype(int)

    # Create data directory if it doesn't exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save results to CSV
    csv_filename = os.path.join(data_dir, 'result.csv')
    df_results.to_csv(csv_filename, index=False)
    print("Results saved to " + csv_filename)

    # Print results to console
    print("\nCMB E-mode polarization power spectrum D_l^EE = l(l+1)C_l^EE/(2pi) [muK^2]:")
    # Set pandas display options for detailed printing
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)  # Adjust width as needed
    print(df_results)


if __name__ == '__main__':
    calculate_cmb_ee_spectrum()
