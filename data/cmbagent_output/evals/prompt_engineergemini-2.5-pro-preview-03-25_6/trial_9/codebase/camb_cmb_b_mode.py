# filename: codebase/camb_cmb_b_mode.py
import os
import camb  # type: ignore
import numpy as np
import pandas as pd

# Ensure the 'data' directory exists for output files
# This should be created by the environment or a setup script,
# but we ensure it here for standalone execution.
if not os.path.exists('data'):
    os.makedirs('data')


def calculate_cmb_b_mode_power_spectrum():
    r"""
    Calculates the CMB raw B-mode polarization power spectrum (C_l^BB)
    for a flat Lambda CDM cosmology using specified parameters with CAMB.

    The B-mode power spectrum (C_l^BB) is computed in units of muK^2
    for multipole moments from l=2 to l=3000.

    The results are saved in a CSV file named 'result.csv' in the 'data/'
    directory, with two columns: 'l' (multipole moment) and 'BB'
    (B-mode power spectrum in muK^2).

    Cosmological Parameters:
    - Hubble constant (H0): 67.5 km/s/Mpc
    - Baryon density (ombh2): 0.022
    - Cold dark matter density (omch2): 0.122
    - Neutrino mass sum (mnu): 0.06 eV
    - Curvature (omk): 0
    - Optical depth to reionization (tau): 0.06
    - Tensor-to-scalar ratio (r): 0.1
    - Scalar amplitude (As): 2e-9
    - Scalar spectral index (ns): 0.965
    """

    # Cosmological parameters
    H0_param = 67.5  # Hubble constant in km/s/Mpc
    ombh2_param = 0.022  # Baryon density * h^2
    omch2_param = 0.122  # Cold dark matter density * h^2
    mnu_param = 0.06  # Sum of neutrino masses in eV
    omk_param = 0.0  # Curvature parameter Omega_k
    tau_param = 0.06  # Optical depth to reionization
    r_tensor_to_scalar_param = 0.1  # Tensor-to-scalar ratio at k_pivot=0.05 Mpc^-1
    As_scalar_amp_param = 2e-9  # Scalar amplitude at k_pivot=0.05 Mpc^-1
    ns_scalar_index_param = 0.965  # Scalar spectral index at k_pivot=0.05 Mpc^-1
    lmax_calc_param = 3000  # Maximum multipole moment for calculation

    # Initialize CAMB parameters object
    pars = camb.CAMBparams()

    # Set cosmological parameters
    # pars.set_cosmology uses H0, ombh2, omch2, mnu, omk, tau
    pars.set_cosmology(H0=H0_param, ombh2=ombh2_param, omch2=omch2_param, 
                       mnu=mnu_param, omk=omk_param, tau=tau_param)

    # Set initial power spectrum parameters
    # pars.InitPower.set_params uses As, ns, r
    # As is scalar amplitude, ns is scalar spectral index, r is tensor-to-scalar ratio
    pars.InitPower.set_params(As=As_scalar_amp_param, ns=ns_scalar_index_param, r=r_tensor_to_scalar_param)

    # Set calculation options
    # We need tensor modes for primordial B-modes.
    pars.WantTensors = True
    
    # Set lmax for calculation.
    # CAMB calculates spectra up to pars.max_l (which is set by lmax here).
    # lens_potential_accuracy is relevant for lensed spectra; setting it to 1 is good practice.
    pars.set_for_lmax(lmax=lmax_calc_param, lens_potential_accuracy=1)

    # Get results from CAMB
    results = camb.get_results(pars)

    # Get CMB power spectra
    # We want raw C_l values (raw_cl=True) in muK^2 units (CMB_unit='muK').
    # 'powers' is a dictionary. For example, powers['tensor'] contains the
    # TT, EE, BB, TE power spectra from tensor modes.
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)

    # Extract the B-mode power spectrum from tensor modes (primordial B-modes)
    # powers['tensor'] is an array of shape (lmax_calc_param + 1, 4).
    # The columns correspond to TT, EE, BB, TE spectra.
    # We need the BB component, which is at index 2.
    # This gives C_l^BB in muK^2.
    cl_bb_tensor = powers['tensor'][:, 2]

    # Define the range of multipole moments (l) required for the output: from 2 to 3000.
    # The array cl_bb_tensor is indexed from l=0.
    # So, for l=2 up to l=lmax_calc_param, we need to slice from index 2 to lmax_calc_param.
    # np.arange(start, stop) goes up to stop-1. So, stop should be lmax_calc_param + 1.
    ls_output = np.arange(2, lmax_calc_param + 1)  # l values from 2 to 3000
    
    # Select the C_l^BB values for the desired range of l.
    # cl_bb_tensor[l_index] gives the value for multipole l_index.
    # So, for l=2, index is 2. For l=lmax_calc_param, index is lmax_calc_param.
    cl_bb_values_output = cl_bb_tensor[2:(lmax_calc_param + 1)]

    # Create a Pandas DataFrame for the results
    df_results = pd.DataFrame({'l': ls_output, 'BB': cl_bb_values_output})

    # Define the output directory and filename
    # The 'data' directory is created at the beginning of the script.
    output_dir = 'data'
    csv_filename = os.path.join(output_dir, 'result.csv')

    # Save the DataFrame to a CSV file, without the pandas index.
    df_results.to_csv(csv_filename, index=False)

    print("CMB B-mode power spectrum calculation complete.")
    print("Results saved to: " + str(csv_filename))
    
    # Print some of the results to the console for verification
    # Using .to_string() to ensure the DataFrame head/tail print fully and clearly.
    print("\nFirst 5 rows of the B-mode power spectrum data (l, BB [muK^2]):")
    print(df_results.head().to_string())
    print("\nLast 5 rows of the B-mode power spectrum data (l, BB [muK^2]):")
    print(df_results.tail().to_string())
    
    # Print summary statistics of the calculated C_l^BB values
    print("\nSummary statistics for C_l^BB [muK^2]:")
    # Using string concatenation for printing numerical results.
    print("Min C_l^BB: " + str(df_results['BB'].min()))
    print("Max C_l^BB: " + str(df_results['BB'].max()))
    print("Mean C_l^BB: " + str(df_results['BB'].mean()))


if __name__ == '__main__':
    calculate_cmb_b_mode_power_spectrum()