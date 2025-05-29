# filename: codebase/compute_angular_diameter_distance.py
import camb
import numpy as np
import pandas as pd
import os

def compute_angular_diameter_distance():
    r"""
    Compute the angular diameter distance d_A (in Mpc) for a flat Lambda CDM cosmology
    using CAMB, for 100 evenly spaced redshift points from z=0 to z=4, and save the results
    to a CSV file.

    Cosmological parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (Omega_b h^2): 0.022
        - Cold dark matter density (Omega_c h^2): 0.122
        - Neutrino mass sum (Sigma m_nu): 0.06 eV
        - Curvature (Omega_k): 0
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (A_s): 2e-9
        - Scalar spectral index (n_s): 0.965

    Output:
        - CSV file 'data/result.csv' with columns:
            - z: Redshift (dimensionless)
            - d_A: Angular diameter distance (Mpc)
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cosmological parameters
    H0 = 67.5  # [km/s/Mpc]
    ombh2 = 0.022  # [dimensionless]
    omch2 = 0.122  # [dimensionless]
    mnu = 0.06  # [eV]
    omk = 0.0  # [dimensionless]
    tau = 0.06  # [dimensionless]
    As = 2e-9  # [dimensionless]
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
        WantTensors=False,
        WantVectors=False,
        WantDerivedParameters=True
    )

    # Calculate background cosmology results
    results = camb.get_background(pars)

    # Define redshift range
    redshifts = np.linspace(0, 4, 100)  # [dimensionless]

    # Calculate angular diameter distance (Mpc)
    d_A = results.angular_diameter_distance(redshifts)  # [Mpc]

    # Save results to CSV
    df = pd.DataFrame({'z': redshifts, 'd_A': d_A})
    csv_path = os.path.join(output_dir, "result.csv")
    df.to_csv(csv_path, index=False)

    # Print summary to console
    pd.set_option("display.precision", 8)
    pd.set_option("display.width", 120)
    print("Angular diameter distance (d_A) [Mpc] for 100 redshifts from z=0 to z=4 (flat Lambda CDM, CAMB):")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
    print("\nResults saved to " + csv_path)


if __name__ == "__main__":
    compute_angular_diameter_distance()