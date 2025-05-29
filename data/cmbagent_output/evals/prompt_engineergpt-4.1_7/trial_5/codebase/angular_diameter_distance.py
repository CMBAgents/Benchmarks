# filename: codebase/angular_diameter_distance.py
import numpy as np
import pandas as pd
import os

from camb import model, initialpower, get_background

def compute_angular_diameter_distance():
    r"""
    Compute the angular diameter distance d_A(z) for a flat Lambda CDM cosmology
    using CAMB, for 100 evenly spaced redshift points from z=0 to z=4.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns 'z' (redshift, dimensionless) and 'd_A' (angular diameter distance in Mpc).
    """
    # Cosmological parameters (units in comments)
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density parameter [dimensionless]
    omch2 = 0.122  # Cold dark matter density parameter [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    As = 2e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]

    # Set up CAMB parameters
    pars = model.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(0, lens_potential_accuracy=0)

    # Redshift array
    z_arr = np.linspace(0, 4, 100)  # Redshift (dimensionless)

    # Get background results
    results = get_background(pars)

    # Compute angular diameter distance for each z
    d_A_arr = np.array([results.angular_diameter_distance(z) for z in z_arr])  # [Mpc]

    # Prepare DataFrame
    df = pd.DataFrame({'z': z_arr, 'd_A': d_A_arr})

    return df

def save_results_to_csv(df, filename):
    r"""
    Save the DataFrame to a CSV file in the data/ directory.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with results.
    filename : str
        Name of the CSV file (should include .csv extension).
    """
    if not os.path.exists('data'):
        os.makedirs('data')
    filepath = os.path.join('data', filename)
    df.to_csv(filepath, index=False)
    print("Results saved to " + filepath)

def print_results_summary(df):
    r"""
    Print a summary of the results to the console.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with results.
    """
    pd.set_option('display.precision', 6)
    pd.set_option('display.width', 120)
    print("\nFirst 5 rows of the angular diameter distance results (z, d_A [Mpc]):")
    print(df.head())
    print("\nLast 5 rows of the angular diameter distance results (z, d_A [Mpc]):")
    print(df.tail())
    print("\nMinimum d_A: " + str(df['d_A'].min()) + " Mpc at z = " + str(df.loc[df['d_A'].idxmin(), 'z']))
    print("Maximum d_A: " + str(df['d_A'].max()) + " Mpc at z = " + str(df.loc[df['d_A'].idxmax(), 'z']))


if __name__ == "__main__":
    df = compute_angular_diameter_distance()
    save_results_to_csv(df, "result.csv")
    print_results_summary(df)