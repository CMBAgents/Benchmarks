# filename: codebase/angular_diameter_distance.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model

def compute_angular_diameter_distance():
    r"""
    Computes the angular diameter distance d_A(z) for a flat Lambda CDM cosmology
    using CAMB for 100 evenly spaced redshift points from z=0 to z=4.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns 'z' (redshift) and 'd_A' (angular diameter distance in Mpc).
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density parameter [dimensionless]
    omch2 = 0.122  # Cold dark matter density parameter [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    As = 2e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]

    # Redshift array
    z_arr = np.linspace(0, 4, 100)  # 100 points from 0 to 4

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=z_arr, kmax=2.0)
    pars.NonLinear = model.NonLinear_none

    # Calculate results for these parameters
    results = camb.get_results(pars)

    # Get angular diameter distance in Mpc for each z
    d_A_arr = np.array([results.angular_diameter_distance(0.0, z) for z in z_arr])  # [Mpc]

    # Prepare DataFrame
    df = pd.DataFrame({'z': z_arr, 'd_A': d_A_arr})

    return df

def save_results_to_csv(df, filename):
    r"""
    Saves the DataFrame to a CSV file in the data/ directory.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with results to save.
    filename : str
        Name of the CSV file (should include .csv extension).
    """
    if not os.path.exists('data'):
        os.makedirs('data')
    filepath = os.path.join('data', filename)
    df.to_csv(filepath, index=False)
    print("Results saved to " + filepath)

def print_results(df):
    r"""
    Prints the angular diameter distance results to the console in a detailed manner.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns 'z' and 'd_A'.
    """
    pd.set_option('display.precision', 6)
    pd.set_option('display.width', 120)
    pd.set_option('display.max_rows', 100)
    print("\nAngular diameter distance d_A(z) [Mpc] for 100 redshift points from z=0 to z=4:\n")
    print(df)
    print("\nMinimum d_A: " + str(df['d_A'].min()) + " Mpc at z = " + str(df.loc[df['d_A'].idxmin(), 'z']))
    print("Maximum d_A: " + str(df['d_A'].max()) + " Mpc at z = " + str(df.loc[df['d_A'].idxmax(), 'z']))


if __name__ == "__main__":
    df = compute_angular_diameter_distance()
    save_results_to_csv(df, "result.csv")
    print_results(df)