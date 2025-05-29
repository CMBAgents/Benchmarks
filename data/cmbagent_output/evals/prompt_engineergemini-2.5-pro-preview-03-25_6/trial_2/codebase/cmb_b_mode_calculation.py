# filename: codebase/cmb_b_mode_calculation.py
import camb
import numpy as np
import pandas as pd
import os


def calculate_cmb_b_mode_spectrum():
    r"""
    Calculates the CMB B-mode polarization power spectrum (Cl^BB) for a flat Lambda CDM cosmology
    using specified parameters with CAMB. Saves the results to a CSV file.
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Baryon density
    omch2 = 0.122  # Cold dark matter density
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0  # Curvature density
    tau = 0.06  # Optical depth to reionization
    r_tensor_to_scalar = 0.1  # Tensor-to-scalar ratio
    As_scalar_amp = 2e-9  # Scalar amplitude
    ns_scalar_index = 0.965  # Scalar spectral index
    
    lmax_calc = 3000  # Maximum multipole moment to calculate

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As_scalar_amp, ns=ns_scalar_index, r=r_tensor_to_scalar)
    
    # We want tensor modes for B-mode calculation
    pars.WantTensors = True
    
    # Set lmax for the calculation
    # lens_potential_accuracy=0 as we are interested in primordial tensor modes, not lensing effects here.
    pars.set_for_lmax(lmax_calc, lens_potential_accuracy=0)

    # Get results from CAMB
    results = camb.get_results(pars)

    # Get CMB power spectra
    # We want raw Cls (raw_cl=True) in muK^2 units (CMB_unit='muK')
    # We are interested in the 'tensor' contribution to the power spectra
    powers = results.get_cmb_power_spectra(pars, lmax=lmax_calc, spectra=['tensor'], CMB_unit='muK', raw_cl=True)

    # powers['tensor'] is an array of shape (4, lmax_calc + 1)
    # The rows are TT, EE, BB, TE. Index 2 corresponds to BB.
    # This gives Cl^BB_tensor for l from 0 to lmax_calc
    cl_bb_tensor_all = powers['tensor'][2, :]  # Units: muK^2

    # We need l from 2 to 3000
    l_values = np.arange(2, lmax_calc + 1)  # Multipole moments
    
    # Extract the corresponding Cl^BB values
    # cl_bb_tensor_all is indexed from l=0. So, cl_bb_tensor_all[l] is Cl_l^BB.
    # We need values from index 2 up to lmax_calc (inclusive).
    bb_values = cl_bb_tensor_all[2 : lmax_calc + 1]  # Units: muK^2

    # Create a Pandas DataFrame
    df_results = pd.DataFrame({'l': l_values, 'BB': bb_values})

    # Define output directory and filename
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_path = os.path.join(output_dir, "result.csv")

    # Save to CSV
    df_results.to_csv(file_path, index=False)

    print("CMB B-mode power spectrum (Cl^BB) calculation complete.")
    print("Results saved to: " + str(file_path))
    print("The file contains " + str(len(df_results)) + " rows, for l from 2 to " + str(lmax_calc) + ".")
    print("Columns: 'l' (multipole moment), 'BB' (Cl^BB in muK^2)")
    
    # Print sample data
    print("\nSample data (first 5 rows):")
    # Temporarily set pandas display options for better console output
    with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 1000):
        print(df_results.head())
    
    print("\nSample data (last 5 rows):")
    with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 1000):
        print(df_results.tail())


if __name__ == '__main__':
    calculate_cmb_b_mode_spectrum()