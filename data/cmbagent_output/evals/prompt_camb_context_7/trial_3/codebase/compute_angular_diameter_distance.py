# filename: codebase/compute_angular_diameter_distance.py
import camb
import numpy as np
import pandas as pd
import os

def compute_angular_diameter_distance():
    r"""
    Compute the angular diameter distance d_A (in Mpc) for 100 evenly spaced redshifts from z=0 to z=4
    using a flat Lambda CDM cosmology with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (Omega_b h^2): 0.022
        - Cold dark matter density (Omega_c h^2): 0.122
        - Neutrino mass sum (Sigma m_nu): 0.06 eV
        - Curvature (Omega_k): 0
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (A_s): 2e-9
        - Scalar spectral index (n_s): 0.965

    The results are saved to 'data/result.csv' with columns:
        - z: Redshift (dimensionless)
        - d_A: Angular diameter distance (Mpc)
    """
    # Cosmological parameters (units in comments)
    H0 = 67.5           # Hubble constant [km/s/Mpc]
    ombh2 = 0.022       # Omega_b h^2 [dimensionless]
    omch2 = 0.122       # Omega_c h^2 [dimensionless]
    mnu = 0.06          # Sum of neutrino masses [eV]
    omk = 0.0           # Curvature Omega_k [dimensionless]
    tau = 0.06          # Optical depth to reionization [dimensionless]
    As = 2e-9           # Scalar amplitude [dimensionless]
    ns = 0.965          # Scalar spectral index [dimensionless]

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

    # Get background cosmology results
    results = camb.get_background(pars)

    # Generate 100 evenly spaced redshift points from 0 to 4
    redshifts = np.linspace(0, 4, 100)

    # Compute angular diameter distance (in Mpc) for each redshift
    d_A = results.angular_diameter_distance(redshifts)

    # Prepare results as a DataFrame
    df = pd.DataFrame({'z': redshifts, 'd_A': d_A})

    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    df.to_csv(output_path, index=False)

    # Print summary to console
    pd.set_option("display.precision", 8)
    print("Angular diameter distance d_A (in Mpc) for 100 redshifts from z=0 to z=4 has been saved to " + output_path)
    print("First 10 rows of the results:")
    print(df.head(10))


if __name__ == "__main__":
    compute_angular_diameter_distance()