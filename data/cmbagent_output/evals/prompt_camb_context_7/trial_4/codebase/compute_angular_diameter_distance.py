# filename: codebase/compute_angular_diameter_distance.py
import camb
import numpy as np
import pandas as pd
import os

def compute_angular_diameter_distance():
    r"""
    Compute the angular diameter distance d_A(z) for a flat Lambda CDM cosmology
    using CAMB, for 100 evenly spaced redshifts from z=0 to z=4, and save the results
    to a CSV file.

    The cosmological parameters used are:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (Omega_b h^2): 0.022
        - Cold dark matter density (Omega_c h^2): 0.122
        - Neutrino mass sum (Sigma m_nu): 0.06 eV
        - Curvature (Omega_k): 0
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (A_s): 2e-9
        - Scalar spectral index (n_s): 0.965

    The output CSV file will have columns:
        - z: Redshift (dimensionless)
        - d_A: Angular diameter distance (Mpc)
    """
    # Output directory and filename
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_csv = os.path.join(output_dir, "result.csv")

    # Cosmological parameters
    H0 = 67.5         # Hubble constant [km/s/Mpc]
    ombh2 = 0.022     # Baryon density Omega_b h^2 [dimensionless]
    omch2 = 0.122     # Cold dark matter density Omega_c h^2 [dimensionless]
    mnu = 0.06        # Sum of neutrino masses [eV]
    omk = 0.0         # Curvature Omega_k [dimensionless]
    tau = 0.06        # Optical depth to reionization [dimensionless]
    As = 2e-9         # Scalar amplitude [dimensionless]
    ns = 0.965        # Scalar spectral index [dimensionless]

    # Set up CAMB parameters
    pars = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu,
        omk=omk,
        tau=tau,
        As=As,
        ns=ns,
        WantScalars=True,
        WantTensors=False,
        WantVectors=False,
        WantDerivedParameters=True
    )

    # Calculate background cosmology
    results = camb.get_background(pars)

    # Redshift array: 100 evenly spaced points from 0 to 4
    z = np.linspace(0, 4, 100)  # dimensionless

    # Compute angular diameter distance [Mpc]
    d_A = results.angular_diameter_distance(z)  # [Mpc]

    # Save results to CSV
    df = pd.DataFrame({'z': z, 'd_A': d_A})
    df.to_csv(output_csv, index=False)

    # Print summary to console
    pd.set_option("display.precision", 8)
    pd.set_option("display.width", 120)
    print("Angular diameter distance d_A(z) [Mpc] for flat Lambda CDM cosmology (CAMB):")
    print("Cosmological parameters:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  Sum m_nu = " + str(mnu) + " eV")
    print("  Omega_k = " + str(omk))
    print("  tau = " + str(tau))
    print("  A_s = " + str(As))
    print("  n_s = " + str(ns))
    print("\nFirst 5 rows of results:")
    print(df.head())
    print("\nLast 5 rows of results:")
    print(df.tail())
    print("\nResults saved to " + output_csv)

if __name__ == "__main__":
    compute_angular_diameter_distance()
