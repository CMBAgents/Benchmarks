# filename: codebase/compute_angular_diameter_distance.py
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
    z : np.ndarray
        Redshift array (dimensionless).
    d_A : np.ndarray
        Angular diameter distance array (Mpc).
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
    z = np.linspace(0, 4, 100)  # 100 points from 0 to 4

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=z, kmax=2.0)
    pars.set_dark_energy(w=-1.0, wa=0.0, dark_energy_model='fluid')

    # Calculate results for these parameters
    results = camb.get_results(pars)

    # Get comoving radial distance (chi) in Mpc for each z
    chi = results.comoving_radial_distance(z)  # [Mpc]

    # Angular diameter distance: d_A(z) = chi(z) / (1 + z) for flat universe
    d_A = chi / (1.0 + z)  # [Mpc]

    return z, d_A

def save_results_to_csv(z, d_A, filename):
    r"""
    Save the redshift and angular diameter distance arrays to a CSV file.

    Parameters
    ----------
    z : np.ndarray
        Redshift array (dimensionless).
    d_A : np.ndarray
        Angular diameter distance array (Mpc).
    filename : str
        Output CSV filename.
    """
    df = pd.DataFrame({'z': z, 'd_A': d_A})
    df.to_csv(filename, index=False)
    print("Results saved to " + filename)
    print("First 10 rows of the results:")
    print(df.head(10).to_string(index=False, float_format="%.6f"))
    print("\nLast 10 rows of the results:")
    print(df.tail(10).to_string(index=False, float_format="%.6f"))
    print("\nTotal number of rows: " + str(len(df)))


if __name__ == "__main__":
    # Ensure data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Compute angular diameter distance
    z, d_A = compute_angular_diameter_distance()

    # Save to CSV
    output_csv = os.path.join(data_dir, "result.csv")
    save_results_to_csv(z, d_A, output_csv)