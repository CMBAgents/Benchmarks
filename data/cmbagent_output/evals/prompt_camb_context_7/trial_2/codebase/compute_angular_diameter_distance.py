# filename: codebase/compute_angular_diameter_distance.py
import camb
import numpy as np
import pandas as pd
import os

def compute_angular_diameter_distance():
    r"""
    Compute the angular diameter distance d_A (in Mpc) for 100 evenly spaced redshifts from z=0 to z=4
    using a flat Lambda CDM cosmology with specified parameters, and save the results to a CSV file.

    Cosmological parameters:
        Hubble constant (H0): 67.5 km/s/Mpc
        Baryon density (Omega_b h^2): 0.022
        Cold dark matter density (Omega_c h^2): 0.122
        Neutrino mass sum (Sigma m_nu): 0.06 eV
        Curvature (Omega_k): 0
        Optical depth to reionization (tau): 0.06
        Scalar amplitude (A_s): 2e-9
        Scalar spectral index (n_s): 0.965

    The results are saved in 'data/result.csv' with columns:
        z   : Redshift (dimensionless)
        d_A : Angular diameter distance (Mpc)
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
        WantDerivedParameters=True,
    )

    # Get background cosmology results
    results = camb.get_background(pars)

    # Generate redshift points
    redshifts = np.linspace(0, 4, 100)  # [dimensionless]

    # Calculate angular diameter distance (in Mpc)
    d_A = results.angular_diameter_distance(redshifts)  # [Mpc]

    # Store results in a pandas DataFrame
    df = pd.DataFrame({'z': redshifts, 'd_A': d_A})

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    df.to_csv(output_path, index=False)

    # Print results to console
    print("Angular diameter distance d_A (in Mpc) for 100 redshifts from z=0 to z=4 has been saved to " + output_path)
    print("First 10 rows of the results:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    compute_angular_diameter_distance()