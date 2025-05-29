# filename: codebase/angular_diameter_distance.py
import numpy as np
import pandas as pd
import os

import camb
from camb import model

def compute_angular_diameter_distance(
    z_array,
    H0=67.5,  # km/s/Mpc
    ombh2=0.022,
    omch2=0.122,
    mnu=0.06,  # eV
    omk=0.0,
    tau=0.06,
    As=2e-9,
    ns=0.965
):
    r"""
    Compute the angular diameter distance d_A(z) for a given array of redshifts using CAMB.

    Parameters
    ----------
    z_array : array_like
        Array of redshift values (dimensionless).
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter Omega_b h^2.
    omch2 : float
        Physical cold dark matter density parameter Omega_c h^2.
    mnu : float
        Total neutrino mass in eV.
    omk : float
        Curvature parameter Omega_k (dimensionless).
    tau : float
        Optical depth to reionization (dimensionless).
    As : float
        Scalar amplitude (dimensionless).
    ns : float
        Scalar spectral index (dimensionless).

    Returns
    -------
    d_A : ndarray
        Angular diameter distance at each redshift in Mpc.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=z_array, kmax=2.0)
    results = camb.get_results(pars)
    # Angular diameter distance in Mpc for each z
    d_A = np.array([results.angular_diameter_distance(0.0, z) for z in z_array])
    return d_A

def main():
    # Ensure data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Redshift array: 100 points from 0 to 4
    z = np.linspace(0, 4, 100)
    # Compute angular diameter distance in Mpc
    d_A = compute_angular_diameter_distance(z)

    # Save to CSV
    df = pd.DataFrame({"z": z, "d_A": d_A})
    csv_path = os.path.join(data_dir, "result.csv")
    df.to_csv(csv_path, index=False)

    # Print results to console in a detailed manner
    np.set_printoptions(precision=6, suppress=True)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.float_format", lambda x: "%.6f" % x)
    print("\nAngular diameter distance d_A(z) [Mpc] for flat Lambda CDM cosmology (100 points from z=0 to z=4):\n")
    print(df)
    print("\nResults saved to " + csv_path)


if __name__ == "__main__":
    main()
