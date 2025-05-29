# filename: codebase/calculate_cmb_b_mode_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def calculate_cmb_b_mode_spectrum():
    r"""
    Calculates the CMB B-mode polarization power spectrum D_l^BB = l(l+1)C_l^BB/(2pi)
    for a flat Lambda CDM cosmology using CAMB and saves the results to a CSV file.

    The cosmological parameters are:
    H0: Hubble constant (km/s/Mpc)
    ombh2: Baryon density
    omch2: Cold dark matter density
    mnu: Sum of neutrino masses (eV)
    omk: Curvature density
    tau: Optical depth to reionization
    r_tensor: Tensor-to-scalar ratio
    As_scalar: Scalar amplitude
    ns_scalar: Scalar spectral index
    lmax_val: Maximum multipole moment
    """

    # Cosmological parameters
    H0_val = 67.5  # H0 (km/s/Mpc)
    ombh2_val = 0.022  # Omega_b h^2
    omch2_val = 0.122  # Omega_c h^2
    mnu_val = 0.06  # Sum of neutrino masses (eV)
    omk_val = 0.0  # Curvature Omega_k
    tau_val = 0.06  # Optical depth to reionization (tau)
    r_tensor_val = 0.0  # Tensor-to-scalar ratio (r)
    As_scalar_val = 2e-9  # Scalar amplitude (A_s)
    ns_scalar_val = 0.965  # Scalar spectral index (n_s)
    lmax_val = 3000 # Max multipole moment l

    # Print parameters being used
    print("Calculating CMB B-mode spectrum with the following parameters:")
    print("H0: " + str(H0_val) + " km/s/Mpc")
    print("ombh2: " + str(ombh2_val))
    print("omch2: " + str(omch2_val))
    print("mnu: " + str(mnu_val) + " eV")
    print("omk: " + str(omk_val))
    print("tau: " + str(tau_val))
    print("r (tensor-to-scalar ratio): " + str(r_tensor_val))
    print("As (scalar amplitude): " + str(As_scalar_val))
    print("ns (scalar spectral index): " + str(ns_scalar_val))
    print("lmax: " + str(lmax_val))
    print("\n")

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0_val, ombh2=ombh2_val, omch2=omch2_val, mnu=mnu_val, omk=omk_val, tau=tau_val)
    pars.InitPower.set_params(As=As_scalar_val, ns=ns_scalar_val, r=r_tensor_val)
    
    # We want lensed CMB spectra, which will generate B-modes from lensing even if r=0
    # CAMB calculates TT, EE, BB, TE. For r=0, BB is from lensing.
    # Set lmax for the calculation.
    # lens_potential_accuracy=1 is a reasonable default.
    # Set WantTensors to True if r > 0. For r=0, B-modes are purely from lensing of E-modes.
    # CAMB's get_lensed_scalar_cls computes lensed Cls.
    pars.set_for_lmax(lmax_val, lens_potential_accuracy=1)

    # Get results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra D_l = l(l+1)C_l/(2pi)
    # This returns spectra for TT, EE, BB, TE up to lmax.
    # Output is in muK^2
    powers = results.get_lensed_scalar_cls(lmax=lmax_val)  # Shape (lmax_val+1, 4)

    # Extract B-mode power spectrum (D_l^BB)
    # Column 2 (0-indexed) is BB
    # The array `powers` is indexed from l=0 to l=lmax_val.
    # We need l from 2 to lmax_val.
    ls = np.arange(0, lmax_val + 1)
    
    # Filter for l >= 2
    l_filter = (ls >= 2)
    ls_filtered = ls[l_filter]
    
    # BB power spectrum D_l^BB = l(l+1)C_l^BB/(2pi) in muK^2
    # powers[:, 2] corresponds to the BB spectrum for l from 0 to lmax_val
    bb_power_spectrum_filtered = powers[l_filter, 2]

    # Create a pandas DataFrame
    df_results = pd.DataFrame({
        'l': ls_filtered,
        'BB': bb_power_spectrum_filtered  # Units: muK^2
    })

    # Define the data directory and filename
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    file_path = os.path.join(data_dir, 'result.csv')

    # Save to CSV
    df_results.to_csv(file_path, index=False)

    print("CMB B-mode power spectrum calculation complete.")
    print("Results saved to: " + str(file_path))
    
    # Print a sample of the data
    print("\nFirst 5 rows of the B-mode power spectrum data:")
    print(df_results.head())
    print("\nLast 5 rows of the B-mode power spectrum data:")
    print(df_results.tail())

if __name__ == '__main__':
    calculate_cmb_b_mode_spectrum()
