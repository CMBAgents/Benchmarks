# filename: codebase/camb_cmb_power_spectrum.py
import os
import camb
import numpy as np
import pandas as pd

def calculate_cmb_power_spectrum(h0, ombh2, omch2, mnu, omk, tau, As, ns, lmax):
    r"""
    Calculates the CMB temperature power spectrum C_l^TT using CAMB.

    Parameters:
    h0 (float): Hubble constant in km/s/Mpc.
    ombh2 (float): Baryon density parameter Omega_b * h^2.
    omch2 (float): Cold dark matter density parameter Omega_c * h^2.
    mnu (float): Sum of neutrino masses in eV.
    omk (float): Curvature density parameter Omega_k.
    tau (float): Optical depth to reionization.
    As (float): Scalar fluctuation amplitude.
    ns (float): Scalar spectral index.
    lmax (int): Maximum multipole moment l.

    Returns:
    pandas.DataFrame: DataFrame with two columns: 'l' (multipole moment)
                      and 'TT' (C_l^TT in muK^2).
                      Contains values for l from 2 to lmax.
    """
    # Initialize CAMB parameters
    pars = camb.CAMBparams()

    # Set cosmological parameters
    pars.set_cosmology(H0=h0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)

    # Set initial power spectrum parameters
    pars.InitPower.set_params(As=As, ns=ns)

    # Set lmax for calculation
    # lens_potential_accuracy=0 means no lensing calculation if only unlensed spectra are needed.
    pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=0)
    
    # We want scalar power spectra
    pars.WantScalars = True
    pars.WantTensors = False  # Not requested, and no r parameter given

    # Get results
    results = camb.get_results(pars)

    # Get CMB power spectra
    # CMB_unit='muK' and raw_cl=True gives C_l in muK^2
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)

    # Extract unlensed scalar TT power spectrum
    # powers['unlensed_scalar'] is an array with columns: TT, EE, BB, TE
    # We need the first column (TT)
    cl_TT_unlensed = powers['unlensed_scalar'][:, 0]  # Units: muK^2

    # Create array of multipole moments l
    ls = np.arange(cl_TT_unlensed.shape[0])  # ls from 0 to lmax

    # Filter for l from 2 to lmax, inclusive
    l_values = ls[2:lmax + 1]
    cl_TT_values = cl_TT_unlensed[2:lmax + 1]

    # Create pandas DataFrame
    df = pd.DataFrame({'l': l_values, 'TT': cl_TT_values})
    df['l'] = df['l'].astype(int)

    return df

def main():
    r"""
    Main function to set parameters, calculate power spectrum, and save results.
    """
    # Cosmological parameters from the problem description
    H0 = 70.0  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Baryon density Omega_b * h^2
    omch2 = 0.122  # Cold dark matter density Omega_c * h^2
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature density Omega_k (0 for flat)
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude A_s
    ns = 0.965  # Scalar spectral index n_s
    lmax_calc = 3000  # Maximum multipole moment l

    print("Calculating CMB temperature power spectrum with CAMB...")
    print("Parameters:")
    print("H0 = " + str(H0) + " km/s/Mpc")
    print("ombh2 = " + str(ombh2))
    print("omch2 = " + str(omch2))
    print("mnu = " + str(mnu) + " eV")
    print("omk = " + str(omk))
    print("tau = " + str(tau))
    print("As = " + str(As))
    print("ns = " + str(ns))
    print("lmax = " + str(lmax_calc))
    
    # Calculate power spectrum
    power_spectrum_df = calculate_cmb_power_spectrum(
        h0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk,
        tau=tau, As=As, ns=ns, lmax=lmax_calc
    )

    # Define data directory and filename
    data_dir = 'data'
    filename = 'result.csv'
    filepath = os.path.join(data_dir, filename)

    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Created directory: " + data_dir)

    # Save results to CSV
    power_spectrum_df.to_csv(filepath, index=False)
    print("CMB power spectrum (l, TT in muK^2) saved to: " + filepath)

    # Print head of the DataFrame to console
    print("\nFirst 5 rows of the calculated power spectrum (C_l^TT in muK^2):")
    # Set pandas display options for detailed printing
    pd.set_option('display.float_format', '{:.6e}'.__mod__)
    pd.set_option('display.max_rows', 10)
    print(power_spectrum_df.head())
    
    print("\nLast 5 rows of the calculated power spectrum (C_l^TT in muK^2):")
    print(power_spectrum_df.tail())

    # Print some specific values for verification
    if len(power_spectrum_df) > 0:
        print("\nValue at l=2:")
        print(power_spectrum_df[power_spectrum_df['l'] == 2])
        
        # Check if l=1000 exists before trying to print
        df_l1000 = power_spectrum_df[power_spectrum_df['l'] == 1000]
        if not df_l1000.empty:
            print("\nValue at l=1000:")
            print(df_l1000)
        
        # Check if l=2000 exists before trying to print
        df_l2000 = power_spectrum_df[power_spectrum_df['l'] == 2000]
        if not df_l2000.empty:
            print("\nValue at l=2000:")
            print(df_l2000)
             
        # Check if l=lmax_calc exists before trying to print
        df_lmax = power_spectrum_df[power_spectrum_df['l'] == lmax_calc]
        if not df_lmax.empty:
            print("\nValue at l=" + str(lmax_calc) + ":")
            print(df_lmax)


if __name__ == '__main__':
    main()