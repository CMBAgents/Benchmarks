# filename: codebase/compute_ang_diam_dist.py
import camb
import numpy as np
import pandas as pd
import os

def compute_angular_diameter_distance():
    r"""
    Compute the angular diameter distance d_A(z) for a flat Lambda CDM cosmology
    using CAMB, for 100 evenly spaced redshifts from z=0 to z=4, and save the results
    to a CSV file.

    Cosmological parameters:
        H0: Hubble constant [km/s/Mpc]
        ombh2: Baryon density Omega_b h^2 [dimensionless]
        omch2: Cold dark matter density Omega_c h^2 [dimensionless]
        mnu: Sum of neutrino masses [eV]
        omk: Curvature Omega_k [dimensionless]
        tau: Optical depth to reionization [dimensionless]
        As: Scalar amplitude [dimensionless]
        ns: Scalar spectral index [dimensionless]

    Output:
        CSV file 'data/result.csv' with columns:
            z: Redshift (dimensionless)
            d_A: Angular diameter distance [Mpc]
    """
    # Cosmological parameters
    H0 = 67.5  # [km/s/Mpc]
    ombh2 = 0.022  # [dimensionless]
    omch2 = 0.122  # [dimensionless]
    mnu = 0.06  # [eV]
    omk = 0.0  # [dimensionless]
    tau = 0.06  # [dimensionless]
    As = 2.0e-9  # [dimensionless]
    ns = 0.965  # [dimensionless]

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
        WantDerivedParameters=True,
    )

    # Calculate background cosmology results
    results = camb.get_background(pars)

    # Define redshift range
    redshifts = np.linspace(0, 4, 100)  # [dimensionless]

    # Calculate angular diameter distance [Mpc]
    d_A = results.angular_diameter_distance(redshifts)  # [Mpc]

    # For z=0, d_A should be exactly 0
    if redshifts[0] == 0:
        d_A[0] = 0.0

    # Prepare output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save results to CSV
    output_filename = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({"z": redshifts, "d_A": d_A})
    df.to_csv(output_filename, index=False)

    # Print summary of results
    pd.set_option("display.precision", 8)
    pd.set_option("display.width", 120)
    print("Angular diameter distance d_A(z) [Mpc] for 100 redshifts from z=0 to z=4 (flat Lambda CDM):")
    print("First 5 rows:")
    print(df.head(5))
    print("\nLast 5 rows:")
    print(df.tail(5))
    print("\nResults saved to " + output_filename)


if __name__ == "__main__":
    compute_angular_diameter_distance()