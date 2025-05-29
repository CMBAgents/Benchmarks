# filename: codebase/cmb_power_spectrum.py
import os
import camb
import numpy as np
import pandas as pd

def calculate_and_save_cmb_power_spectrum():
    r"""
    Calculates the CMB temperature power spectrum using CAMB for specified cosmological parameters
    and saves the results to a CSV file.

    The function computes D_l^TT = l(l+1)C_l^TT/(2*pi) in muK^2 for multipole moments
    l from 2 to 3000.

    Cosmological Parameters:
    - H0: 67.5 km/s/Mpc (Hubble constant)
    - ombh2: 0.02 (Baryon density * h^2)
    - omch2: 0.122 (Cold dark matter density * h^2)
    - mnu: 0.06 eV (Sum of neutrino masses)
    - omk: 0 (Curvature density)
    - tau: 0.06 (Optical depth to reionization)
    - As: 2e-9 (Scalar amplitude)
    - ns: 0.965 (Scalar spectral index)
    - lmax: 3000 (Maximum multipole moment)
    """
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Created directory: " + data_dir)

    # Set up CAMB parameters
    pars = camb.CAMBparams()

    # Set cosmological parameters
    h0_val = 67.5  # H0 in km/s/Mpc
    ombh2_val = 0.02  # Omega_b * h^2
    omch2_val = 0.122  # Omega_c * h^2
    mnu_val = 0.06  # Sum of neutrino masses in eV
    omk_val = 0.0  # Omega_k (curvature)
    tau_val = 0.06  # Optical depth to reionization

    pars.set_cosmology(
        H0=h0_val,
        ombh2=ombh2_val,
        omch2=omch2_val,
        mnu=mnu_val,
        omk=omk_val,
        tau=tau_val
    )

    # Set initial power spectrum parameters
    As_val = 2e-9  # Scalar amplitude
    ns_val = 0.965  # Scalar spectral index
    pars.InitPower.set_params(As=As_val, ns=ns_val)

    # Set lmax and lensing accuracy
    lmax_val = 3000
    pars.set_for_lmax(lmax=lmax_val, lens_potential_accuracy=1)

    # We want scalar power spectra
    pars.WantScalars = True 
    # By default, CAMB calculates lensed spectra if lensing is enabled (which it is by default)

    # Get results
    print("Running CAMB to calculate power spectra...")
    results = camb.get_results(pars)
    print("CAMB calculation complete.")

    # Get CMB power spectra D_l = l(l+1)C_l/(2*pi) in muK^2
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=False)
    
    # totCL is an array with columns TT, EE, BB, TE
    totCL = powers['total']
    
    # Multipole moments l
    ls = np.arange(totCL.shape[0])

    # We need l from 2 to 3000
    l_min_req = 2
    l_max_req = 3000
    
    if l_min_req < 0 or l_max_req > lmax_val:
        print("Error: Requested l range [" + str(l_min_req) + ", " + str(l_max_req) + "] is out of computed bounds [0, " + str(lmax_val) + "].")
        return

    l_column = ls[l_min_req : l_max_req + 1]
    tt_column = totCL[l_min_req : l_max_req + 1, 0] 

    # Create pandas DataFrame
    df = pd.DataFrame({
        'l': l_column,
        'TT': tt_column
    })

    # Save to CSV
    output_filename = "result.csv"
    output_path = os.path.join(data_dir, output_filename)
    df.to_csv(output_path, index=False)

    print("CMB temperature power spectrum data saved to: " + output_path)
    print("The 'l' column contains multipole moments.")
    print("The 'TT' column contains D_l^TT = l(l+1)C_l^TT/(2*pi) in muK^2.")
    
    print("\nFirst 5 rows of the data:")
    pd.set_option('display.float_format', '{:.6e}'.format)
    print(df.head())
    
    print("\nLast 5 rows of the data:")
    print(df.tail())

if __name__ == "__main__":
    calculate_and_save_cmb_power_spectrum()