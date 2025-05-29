# filename: codebase/angular_diameter_distance.py
import numpy as np
import pandas as pd
import os

import camb
from camb import model, initialpower

def compute_angular_diameter_distance():
    r"""
    Compute the angular diameter distance d_A(z) for a flat Lambda CDM cosmology using CAMB.

    Returns
    -------
    z : ndarray
        Redshift array (dimensionless).
    d_A : ndarray
        Angular diameter distance array (Mpc).
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Omega_b h^2
    omch2 = 0.122  # Omega_c h^2
    mnu = 0.06  # Neutrino mass sum in eV
    omk = 0.0  # Curvature
    tau = 0.06  # Optical depth
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index

    # Redshift array
    z = np.linspace(0, 4, 100)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=z, kmax=2.0)
    pars.NonLinear = model.NonLinear_none

    # Calculate results for these parameters
    results = camb.get_results(pars)

    # Get angular diameter distance in Mpc for each z
    # d_A(z) = comoving_radial_distance(z) / (1+z)
    d_A = np.array([results.angular_diameter_distance(0, zi) for zi in z])  # Mpc

    return z, d_A

def save_results_to_csv(z, d_A, filename):
    r"""
    Save the redshift and angular diameter distance arrays to a CSV file.

    Parameters
    ----------
    z : ndarray
        Redshift array (dimensionless).
    d_A : ndarray
        Angular diameter distance array (Mpc).
    filename : str
        Output CSV filename.
    """
    df = pd.DataFrame({'z': z, 'd_A': d_A})
    df.to_csv(filename, index=False)
    print("Results saved to " + filename)

def print_results(z, d_A):
    r"""
    Print the redshift and angular diameter distance arrays in a detailed manner.

    Parameters
    ----------
    z : ndarray
        Redshift array (dimensionless).
    d_A : ndarray
        Angular diameter distance array (Mpc).
    """
    pd.set_option("display.max_rows", 100)
    df = pd.DataFrame({'z': z, 'd_A [Mpc]': d_A})
    print("\nAngular diameter distance d_A(z) [Mpc] for 100 redshift points from z=0 to z=4:\n")
    print(df)
    print("\nMinimum d_A: " + str(np.min(d_A)) + " Mpc at z = " + str(z[np.argmin(d_A)]))
    print("Maximum d_A: " + str(np.max(d_A)) + " Mpc at z = " + str(z[np.argmax(d_A)]))


if __name__ == "__main__":
    # Ensure data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Compute angular diameter distance
    z, d_A = compute_angular_diameter_distance()

    # Save results to CSV
    csv_filename = os.path.join(data_dir, "result.csv")
    save_results_to_csv(z, d_A, csv_filename)

    # Print results to console
    print_results(z, d_A)