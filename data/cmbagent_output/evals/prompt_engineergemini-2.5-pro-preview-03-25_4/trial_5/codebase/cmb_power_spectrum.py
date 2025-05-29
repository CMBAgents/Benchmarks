# filename: codebase/cmb_power_spectrum.py
import camb
import numpy as np
import pandas as pd
import os


def calculate_cmb_power_spectrum():
    r"""
    Calculates the CMB E-mode polarization power spectrum for a given cosmology.

    The function uses CAMB to compute the lensed scalar E-mode power spectrum
    D_l^EE = l(l+1)C_l^EE/(2*pi) in units of muK^2 for multipole moments
    l from 2 to 3000. The results are saved to a CSV file.

    Cosmological Parameters:
    - H0 (Hubble constant): 67.5 km/s/Mpc
    - ombh2 (Baryon density Omega_b * h^2): 0.022
    - omch2 (Cold dark matter density Omega_c * h^2): 0.122
    - mnu (Sum of neutrino masses): 0.06 eV
    - omk (Curvature Omega_k): 0
    - tau (Optical depth to reionization): 0.04
    - As (Scalar amplitude): 2e-9
    - ns (Scalar spectral index): 0.965
    - lmax_calc (Maximum multipole moment for calculation): 3000
    """

    # Define cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Physical baryon density
    omch2 = 0.122  # Physical cold dark matter density
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature density
    tau = 0.04  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    lmax_calc = 3000  # Maximum multipole moment

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax_calc, lens_potential_accuracy=1)  # lens_potential_accuracy=1 for lensed spectra (default)

    # Get results from CAMB
    print("Running CAMB to calculate power spectra...")
    results = camb.get_results(pars)
    print("CAMB calculation complete.")

    # Get lensed scalar CMB power spectra D_l = l(l+1)C_l/(2*pi)
    # The output is an array with columns: TT, EE, BB, TE
    # Units are muK^2 when CMB_unit='muK' (default for this function)
    # spectra_data[l,0] is D_l^TT
    # spectra_data[l,1] is D_l^EE
    # spectra_data[l,2] is D_l^BB
    # spectra_data[l,3] is D_l^TE
    spectra_data = results.get_lensed_scalar_cls(CMB_unit='muK', lmax=lmax_calc)
    
    # Multipole moments l, from l=0 to lmax_calc
    ls = np.arange(lmax_calc + 1)

    # We need l from 2 to 3000
    # The array `ls` and `spectra_data` are 0-indexed.
    # So, l=2 corresponds to index 2.
    # We want up to l=3000, which is index 3000.
    # Slice from index 2 up to and including index 3000.
    l_values = ls[2:lmax_calc + 1]  # l values from 2 to 3000
    
    # Extract D_l^EE, which is the second column (index 1)
    EE_power_spectrum = spectra_data[2:lmax_calc + 1, 1]  # D_l^EE in muK^2

    # Create a pandas DataFrame
    df_results = pd.DataFrame({
        'l': l_values,
        'EE': EE_power_spectrum  # D_l^EE in muK^2
    })

    # Ensure the data directory exists
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # Save results to CSV
    file_path = os.path.join(data_dir, "result.csv")
    df_results.to_csv(file_path, index=False)
    print("CMB E-mode power spectrum data saved to: " + file_path)

    # Print head and tail of the DataFrame
    print("\nFirst 5 rows of the E-mode power spectrum data:")
    print(df_results.head())
    print("\nLast 5 rows of the E-mode power spectrum data:")
    print(df_results.tail())
    
    # Print some specific values for verification
    if len(df_results) > 0:
        l_100_idx = df_results[df_results['l'] == 100].index
        if not l_100_idx.empty:
            ee_at_l_100 = df_results.loc[l_100_idx[0], 'EE']
            print("\nE-mode power spectrum D_l^EE at l=100: " + str(ee_at_l_100) + " muK^2")
        
        l_1000_idx = df_results[df_results['l'] == 1000].index
        if not l_1000_idx.empty:
            ee_at_l_1000 = df_results.loc[l_1000_idx[0], 'EE']
            print("E-mode power spectrum D_l^EE at l=1000: " + str(ee_at_l_1000) + " muK^2")


if __name__ == '__main__':
    calculate_cmb_power_spectrum()