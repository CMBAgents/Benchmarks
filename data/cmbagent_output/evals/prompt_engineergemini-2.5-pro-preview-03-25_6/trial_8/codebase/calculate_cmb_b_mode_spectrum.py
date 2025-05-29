# filename: codebase/calculate_cmb_b_mode_spectrum.py
import os
import camb
import pandas as pd
import numpy as np


def calculate_cmb_b_mode_spectrum():
    r"""
    Calculates the CMB raw B-mode polarization power spectrum (C_l^BB)
    for a flat Lambda CDM cosmology using specified parameters with CAMB.

    The B-mode power spectrum (C_l^BB) is computed in units of muK^2
    for multipole moments from l=2 to l=3000.
    The results are saved in a CSV file.
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Baryon density
    omch2 = 0.122  # Cold dark matter density
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0  # Curvature
    tau = 0.06  # Optical depth to reionization
    r_tensor_to_scalar = 0.1  # Tensor-to-scalar ratio
    As_scalar_amplitude = 2e-9  # Scalar amplitude
    ns_scalar_spectral_index = 0.965  # Scalar spectral index
    lmax_calc = 3000  # Maximum multipole moment

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As_scalar_amplitude, ns=ns_scalar_spectral_index, r=r_tensor_to_scalar)
    
    # We want tensor modes for B-modes
    pars.WantTensors = True
    
    # Set calculations for lmax
    # lens_potential_accuracy=0 because we want raw (unlensed) B-modes from tensors
    pars.set_for_lmax(lmax=lmax_calc, lens_potential_accuracy=0)

    # Get results
    results = camb.get_results(pars)

    # Get CMB power spectra
    # CMB_unit='muK' gives C_l in muK^2
    # raw_cl=True gives C_l, False gives l(l+1)C_l/(2pi)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)

    # Extract tensor spectra ( primordial B-modes are in the tensor part)
    # Columns are: l, TT, EE, BB, TE
    # We need l (column 0) and BB (column 3)
    tensor_cls = powers['tensor']
    
    ls = tensor_cls[:, 0].astype(int)  # Multipole moment l
    cl_bb_muk2 = tensor_cls[:, 3]      # C_l^BB in muK^2

    # Create a DataFrame
    # Filter for l from 2 to lmax_calc (CAMB output for polarization starts at l=2)
    # If lmax_calc is 3000, ls will be [2, 3, ..., 3000]
    df_results = pd.DataFrame({'l': ls, 'BB': cl_bb_muk2})
    
    # Ensure l is within the desired range [2, 3000]
    # This step is mostly a safeguard if lmax_calc was set higher for some reason
    # but with lmax_calc=3000, it should already be correct.
    df_results = df_results[(df_results['l'] >= 2) & (df_results['l'] <= lmax_calc)]

    # Define the output directory and file path
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_path = os.path.join(output_dir, "result.csv")

    # Save to CSV
    df_results.to_csv(file_path, index=False)

    # Print information about the results
    print("CMB B-mode power spectrum (C_l^BB) calculation complete.")
    print("First 5 rows of the results:")
    # Configure pandas to print more digits for float values if necessary
    pd.set_option('display.float_format', '{:.6e}'.format)
    print(df_results.head())
    pd.reset_option('display.float_format')  # Reset to default
    print("Shape of the results DataFrame: " + str(df_results.shape))
    print("Results saved to: " + file_path)


if __name__ == '__main__':
    calculate_cmb_b_mode_spectrum()
