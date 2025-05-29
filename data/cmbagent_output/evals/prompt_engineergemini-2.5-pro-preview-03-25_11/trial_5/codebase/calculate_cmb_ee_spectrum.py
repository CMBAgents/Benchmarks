# filename: codebase/calculate_cmb_ee_spectrum.py
import camb
import numpy as np
import pandas as pd
import os
from io import StringIO  # Used for capturing df.info() output

# Code Explanation:
# The code calculates the Cosmic Microwave Background (CMB) E-mode polarization power spectrum
# for a flat Lambda CDM cosmology using specified parameters with the CAMB library.
#
# 1. Cosmological Parameters:
#    - Hubble constant (H0): 67.5 km/s/Mpc
#    - Baryon density (ombh2): 0.022
#    - Cold dark matter density (omch2): 0.122
#    - Optical depth to reionization (tau): 0.1
#    - Scalar spectral index (ns): 0.95
#    - Scalar amplitude (As): 1.8e-9 * exp(2 * tau)
#    - Reionization model: CAMB's default Tanh model is used. The optical depth tau is specified,
#      and the reionization exponent (Reion.ReionizationExponent) is set to 2.0, influencing
#      the shape of the reionization history $x_e(z)$.
#
# 2. CAMB Configuration:
#    - A camb.CAMBparams object is initialized.
#    - `set_cosmology` is used to define H0, ombh2, omch2, and tau. omk=0 for flatness is default.
#    - `InitPower.set_params` sets As and ns.
#    - `Reion.ReionizationExponent` is set to 2.0.
#    - `set_for_lmax` configures CAMB to calculate spectra up to lmax=100.
#      `lens_potential_accuracy=0` ensures unlensed spectra are primary.
#    - `DoLensing` is explicitly set to False.
#    - `WantTensors` is False (default), so only scalar perturbations are considered.
#
# 3. Power Spectrum Calculation:
#    - `camb.get_results` runs the Boltzmann solver.
#    - `results.get_cmb_power_spectra` retrieves the power spectra.
#      The `CMB_unit='muK'` argument ensures output D_l = l(l+1)C_l/(2*pi) is in muK^2.
#    - The E-mode power spectrum (D_l^EE) is extracted from the 'unlensed_scalar' component.
#
# 4. Output:
#    - Multipole moments (l) from 2 to 100 and their corresponding D_l^EE values are selected.
#    - These are stored in a pandas DataFrame with columns 'l' (integer) and 'EE' (float, in muK^2).
#    - The DataFrame is saved to 'data/result.csv'. The 'data/' directory is created if absent.
#    - A summary of the results (head, tail, info, description, and an example value) is printed to the console.

def calculate_cmb_ee_spectrum():
    r"""
    Calculates the CMB E-mode polarization power spectrum D_l^EE = l(l+1)C_l^EE / (2*pi)
    for a specific flat Lambda CDM cosmology using CAMB.

    The cosmological parameters are:
    - H0 = 67.5 km/s/Mpc (Hubble constant)
    - ombh2 = 0.022 (Baryon density parameter * h^2)
    - omch2 = 0.122 (Cold dark matter density parameter * h^2)
    - tau = 0.1 (Optical depth to reionization)
    - ns = 0.95 (Scalar spectral index)
    - As = 1.8e-9 * exp(2 * tau) (Scalar amplitude at k_pivot=0.05 Mpc^-1)
    - Reionization model: Tanh model with ReionizationExponent = 2.0.

    The spectrum D_l^EE is calculated in units of muK^2 for multipole moments l from 2 to 100.
    The results are saved in a CSV file named 'result.csv' under the 'data/' directory.
    """

    # Cosmological parameters
    H0_val = 67.5  # Hubble constant in km/s/Mpc
    ombh2_val = 0.022  # Baryon density * h^2
    omch2_val = 0.122  # Cold dark matter density * h^2
    tau_val = 0.1  # Optical depth to reionization
    ns_val = 0.95  # Scalar spectral index
    
    # Calculate scalar amplitude As
    # As is the primordial scalar amplitude at k_pivot = 0.05 Mpc^-1
    As_val = 1.8e-9 * np.exp(2 * tau_val)

    # Reionization parameters
    reion_exponent_val = 2.0  # Exponent for the Tanh reionization model

    # Maximum multipole moment for output
    lmax_calc = 100  # Calculate spectra up to l=100

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    
    # Set cosmological parameters
    # omk=0 for flat cosmology is CAMB's default.
    # Default neutrino mass (mnu=0.06 eV, sum over three species) is used as not specified.
    pars.set_cosmology(H0=H0_val, ombh2=ombh2_val, omch2=omch2_val, tau=tau_val)
    
    # Set initial power spectrum parameters
    pars.InitPower.set_params(As=As_val, ns=ns_val)
    # Tensor to scalar ratio r=0 is default, so only scalar modes are considered.

    # Set reionization model details
    # pars.Reion.ReionizationModel = 'Tanh' is the default in CAMB.
    # When tau is set via set_cosmology, CAMB uses it as a target for the Tanh model,
    # adjusting ReionizationRedshift internally, using the specified ReionizationExponent.
    pars.Reion.ReionizationExponent = reion_exponent_val
    
    # Set calculation range for l (multipole moment)
    # lens_potential_accuracy=0 means unlensed C_ls will be primary.
    pars.set_for_lmax(lmax=lmax_calc, lens_potential_accuracy=0)
    
    # Ensure we are getting unlensed scalar Cls
    pars.DoLensing = False      # Compute unlensed power spectra
    pars.WantTensors = False    # Do not include tensor mode contributions

    # Get results from CAMB
    results = camb.get_results(pars)

    # Get CMB power spectra
    # powers is a dictionary. For unlensed scalar spectra, the key is 'unlensed_scalar'.
    # The output D_l = l(l+1)C_l/(2*pi) is requested in muK^2 units.
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax_calc)
    
    # Extract D_l^EE (E-mode polarization power spectrum)
    # Column 1 (0-indexed) of 'unlensed_scalar' is D_l^EE.
    # The array `powers['unlensed_scalar']` has shape (lmax_calc + 1, number_of_spectra).
    # dl_EE values are in muK^2.
    dl_EE = powers['unlensed_scalar'][:, 1]

    # Create array of l values (from 0 to lmax_calc)
    l_values = np.arange(lmax_calc + 1)  # Integer multipole moments: 0, 1, ..., lmax_calc

    # Select data for l from 2 to 100 (inclusive)
    # This corresponds to array indices 2 through lmax_calc.
    l_column = l_values[2:lmax_calc + 1]
    EE_column = dl_EE[2:lmax_calc + 1]

    # Create pandas DataFrame
    df_results = pd.DataFrame({
        'l': l_column,    # Multipole moment l
        'EE': EE_column   # D_l^EE in muK^2
    })

    # Create data directory if it doesn't exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Created directory: " + str(data_dir))
    else:
        print("Directory already exists: " + str(data_dir))

    
    # Save results to CSV file
    file_path = os.path.join(data_dir, 'result.csv')
    df_results.to_csv(file_path, index=False, float_format='%.8e')  # Save with specific float format

    print("\nCMB E-mode polarization power spectrum calculation complete.")
    print("Results saved to: " + str(file_path))
    
    # Print summary of the DataFrame to console
    print("\nFirst 5 rows of the results:")
    print(df_results.head().to_string())
    print("\nLast 5 rows of the results:")
    print(df_results.tail().to_string())
    
    print("\nDataFrame Info:")
    buffer = StringIO()
    # Store df.info() output in buffer then print, to ensure it's handled correctly by stdout.
    df_results.info(buf=buffer, verbose=False, memory_usage=False) 
    print(buffer.getvalue())

    print("DataFrame Description:")
    print(df_results.describe().to_string())
    
    # Print one example value from the results for quick check
    if not df_results.empty:
        example_l = df_results['l'].iloc[0]  # First l value in the selection (l=2)
        example_ee = df_results['EE'].iloc[0]  # Corresponding D_l^EE
        print("\nExample from calculated data: For l = " + str(example_l) + ", D_l^EE = " + ("%.6e" % example_ee) + " muK^2")
    else:
        print("\nNo data to show in example (DataFrame is empty).")


if __name__ == '__main__':
    calculate_cmb_ee_spectrum()