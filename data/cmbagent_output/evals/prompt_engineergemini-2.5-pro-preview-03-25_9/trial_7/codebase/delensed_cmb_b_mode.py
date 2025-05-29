# filename: codebase/delensed_cmb_b_mode.py
import camb
import numpy as np
import pandas as pd
import os

def calculate_delensed_cmb_b_mode():
    r"""
    Calculates the delensed CMB B-mode polarization power spectrum (C_ell^BB)
    for a flat Lambda CDM cosmology using CAMB.

    The script sets cosmological parameters, computes CMB power spectra including
    tensor modes and lensing effects, applies a specified delensing efficiency,
    and saves the resulting delensed B-mode spectrum to a CSV file.
    """

    # Cosmological Parameters
    H0_val = 67.5  # Hubble constant (km/s/Mpc)
    ombh2_val = 0.022  # Baryon density
    omch2_val = 0.122  # Cold dark matter density
    mnu_val = 0.06  # Neutrino mass sum (eV)
    omk_val = 0.0  # Curvature
    tau_val = 0.06  # Optical depth to reionization
    r_val = 0.1  # Tensor-to-scalar ratio
    As_val = 2e-9  # Scalar amplitude
    ns_val = 0.965  # Scalar spectral index

    lmax_calc = 3000  # Maximum multipole for calculation and output
    delensing_efficiency = 0.1  # Delensing efficiency (10%)

    print("Calculating delensed CMB B-mode power spectrum with CAMB.")
    print("Cosmological Parameters used:")
    print("H0: " + str(H0_val) + " km/s/Mpc")
    print("ombh2: " + str(ombh2_val))
    print("omch2: " + str(omch2_val))
    print("mnu: " + str(mnu_val) + " eV")
    print("omk: " + str(omk_val))
    print("tau: " + str(tau_val))
    print("r: " + str(r_val))
    print("As: " + str(As_val))
    print("ns: " + str(ns_val))
    print("lmax: " + str(lmax_calc))
    print("Delensing efficiency: " + str(delensing_efficiency))
    print("\n")

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0_val, ombh2=ombh2_val, omch2=omch2_val, mnu=mnu_val, omk=omk_val, tau=tau_val)
    pars.InitPower.set_params(As=As_val, ns=ns_val, r=r_val)
    
    # Set lmax for calculation, ensuring high enough for lensing and tensors
    pars.set_for_lmax(lmax_calc, lens_potential_accuracy=1)

    # We need tensor modes for primordial B-modes and lensed scalars for lensing B-modes
    pars.WantTensors = True
    pars.DoLensing = True

    # Get results
    results = camb.get_results(pars)

    # Get CMB power spectra (C_ell in muK^2)
    # raw_cl=True returns C_ell directly. CMB_unit='muK' ensures units are muK^2.
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)

    # Extract B-mode components
    # powers['tensor'] has columns: TT, EE, BB, TE. BB is index 2.
    # These are C_ell values in muK^2.
    cl_bb_primordial = powers['tensor'][:, 2]
    
    # powers['lensed_scalar'] has columns: TT, EE, BB, TE. BB is index 2.
    # This is the C_ell^BB from lensing of E-modes.
    cl_bb_lensing_component = powers['lensed_scalar'][:, 2]

    # Calculate delensed B-mode power spectrum
    # C_ell^BB_delensed = C_ell^BB_primordial + (1 - efficiency) * C_ell^BB_lensing_component
    cl_bb_delensed = cl_bb_primordial + (1.0 - delensing_efficiency) * cl_bb_lensing_component

    # Prepare data for CSV output
    # Multipoles 'ls' go from 0 to lmax_calc
    ls = np.arange(cl_bb_delensed.shape[0])

    # We need l from 2 to lmax_calc (inclusive)
    l_start_index = 2
    
    # Ensure that l_start_index is valid for the array shapes
    if l_start_index >= len(ls):
        print("Error: l_start_index (" + str(l_start_index) + ") is out of bounds for ls array of length " + str(len(ls)))
        print("This might happen if lmax_calc is too small.")
        return

    ls_output = ls[l_start_index : lmax_calc + 1]
    cl_bb_delensed_output = cl_bb_delensed[l_start_index : lmax_calc + 1]

    # Create Pandas DataFrame
    df_results = pd.DataFrame({
        'l': ls_output.astype(int),
        'BB': cl_bb_delensed_output  # C_ell^BB in muK^2
    })

    # Save to CSV
    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    csv_filename = os.path.join(output_dir, 'result.csv')
    df_results.to_csv(csv_filename, index=False, float_format='%.8e')

    print("Successfully calculated and saved delensed CMB B-mode power spectrum.")
    print("Results saved to: " + str(csv_filename))
    print("\nSummary of the results (delensed C_ell^BB in muK^2):")
    print("DataFrame head:")
    print(df_results.head().to_string())
    print("\nDataFrame tail:")
    print(df_results.tail().to_string())
    print("\nDataFrame description:")
    print(df_results.describe().to_string())


if __name__ == '__main__':
    calculate_delensed_cmb_b_mode()