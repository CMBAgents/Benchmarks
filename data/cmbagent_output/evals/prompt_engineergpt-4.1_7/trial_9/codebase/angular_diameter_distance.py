# filename: codebase/angular_diameter_distance.py
import numpy as np
import pandas as pd
import os

from camb import model, initialpower, get_background

def compute_angular_diameter_distance():
    r""" Compute the angular diameter distance d_A(z) for a flat Lambda CDM cosmology using CAMB, for 100 evenly spaced redshift points from z=0 to z=4.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns:
            - 'z': Redshift (dimensionless)
            - 'd_A': Angular diameter distance (Mpc)
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Baryon density parameter Omega_b h^2
    omch2 = 0.122  # Cold dark matter density parameter Omega_c h^2
    mnu = 0.06  # Neutrino mass sum in eV
    omk = 0.0  # Curvature parameter Omega_k
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index

    # Redshift array
    z_arr = np.linspace(0, 4, 100)

    # Set up CAMB parameters
    pars = model.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_dark_energy(w=-1.0, wa=0.0, dark_energy_model='fluid')
    pars.set_accuracy(AccuracyBoost=1.0)
    pars.set_matter_power(redshifts=z_arr, kmax=2.0)

    # Get background results
    results = get_background(pars)

    # Compute angular diameter distance for each z
    d_A = np.array([results.angular_diameter_distance(0.0, z) for z in z_arr])  # in Mpc

    # Save to DataFrame
    df = pd.DataFrame({'z': z_arr, 'd_A': d_A})

    return df

def main():
    r""" Main function to compute and save the angular diameter distance as a function of redshift. """
    # Ensure data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Compute distances
    df = compute_angular_diameter_distance()

    # Save to CSV
    csv_path = os.path.join(data_dir, "result.csv")
    df.to_csv(csv_path, index=False)

    # Print summary of results
    pd.set_option("display.precision", 6)
    pd.set_option("display.width", 120)
    print("Angular diameter distance d_A(z) [Mpc] for flat Lambda CDM cosmology (CAMB):")
    print("First 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
    print("\nResults saved to " + csv_path)

if __name__ == "__main__":
    main()
