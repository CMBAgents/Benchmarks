# filename: codebase/compute_angular_diameter_distance.py
import numpy as np
import pandas as pd
import os
import camb
from camb import model

def compute_angular_diameter_distance():
    r"""
    Computes the angular diameter distance d_A(z) for 100 evenly spaced redshift points from z=0 to z=4
    using a flat Lambda CDM cosmology with specified parameters, and saves the results to data/result.csv.

    Returns
    -------
    None

    The function prints the results to the console and saves them in a CSV file with columns:
    - z: Redshift (dimensionless)
    - d_A: Angular diameter distance (Mpc)
    """
    # Cosmological parameters (units in comments)
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    As = 2e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]

    # Redshift array
    z_arr = np.linspace(0, 4, 100)  # Redshift (dimensionless)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=z_arr, kmax=2.0)
    pars.NonLinear = model.NonLinear_none

    # Calculate results for background cosmology
    results = camb.get_results(pars)
    # Get angular diameter distance in Mpc for each redshift
    d_A_arr = np.array([results.angular_diameter_distance(0, z) for z in z_arr])  # [Mpc]

    # Prepare DataFrame
    df = pd.DataFrame({'z': z_arr, 'd_A': d_A_arr})

    # Ensure data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save to CSV
    csv_path = os.path.join(data_dir, "result.csv")
    df.to_csv(csv_path, index=False)

    # Print results to console in a detailed manner
    np.set_printoptions(precision=6, suppress=True)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.float_format', lambda x: "%.6f" % x)
    print("Angular diameter distance d_A(z) [Mpc] for flat Lambda CDM cosmology (100 points from z=0 to z=4):\n")
    print(df)
    print("\nResults saved to " + csv_path)

if __name__ == "__main__":
    compute_angular_diameter_distance()
