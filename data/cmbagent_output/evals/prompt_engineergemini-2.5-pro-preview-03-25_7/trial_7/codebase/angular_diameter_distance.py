# filename: codebase/angular_diameter_distance.py
import numpy as np
import pandas as pd
import camb
import os


def calculate_angular_diameter_distance():
    r"""
    Calculates the angular diameter distance (d_A) for a flat Lambda CDM cosmology
    using specified parameters with CAMB.

    The calculation is performed for 100 evenly spaced redshift points from z=0 to z=4.
    The results are saved in a CSV file named 'result.csv' in the 'data/' directory,
    with two columns: 'z' (Redshift) and 'd_A' (Angular diameter distance in Mpc).

    Cosmological Parameters:
        Hubble constant (H_0): 67.5 km/s/Mpc
        Baryon density (Omega_b h^2): 0.022
        Cold dark matter density (Omega_c h^2): 0.122
        Neutrino mass sum (Sigma m_nu): 0.06 eV
        Curvature (Omega_k): 0
        Optical depth to reionization (tau): 0.06
        Scalar amplitude (A_s): 2e-9
        Scalar spectral index (n_s): 0.965
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Baryon density * h^2
    omch2 = 0.122  # Cold dark matter density * h^2
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0  # Curvature
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_background(redshift_max=4.0)  # Ensure background evolution is computed up to z=4

    # Redshift range
    redshifts = np.linspace(0, 4, 100)  # 100 evenly spaced redshift points

    # Calculate angular diameter distances
    # CAMB's angular_diameter_distance function returns d_A in Mpc
    # It can take an array of redshifts
    try:
        # For non-flat cosmologies, angular_diameter_distance(z) is fine.
        # For flat cosmologies, comoving_radial_distance(z) / (1+z) is d_A(z).
        # However, CAMB's angular_diameter_distance function should handle this correctly.
        # Let's get results object first to ensure consistency
        results = camb.get_background(pars)
        angular_distances = results.angular_diameter_distance(redshifts)  # d_A in Mpc
    except Exception as e:
        print("Error during CAMB calculation: " + str(e))
        return

    # Create a DataFrame to store the results
    df_results = pd.DataFrame({'z': redshifts, 'd_A': angular_distances})

    # Define the data directory and ensure it exists
    data_dir = 'data/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save the results to a CSV file
    file_path = os.path.join(data_dir, 'result.csv')
    df_results.to_csv(file_path, index=False)

    print("Angular diameter distances calculated and saved to " + str(file_path))
    print("First few rows of the results:")
    # Set pandas display options to show more precision if needed
    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    print(df_results.head())
    print("\nLast few rows of the results:")
    print(df_results.tail())
    
    # Print some specific values for verification
    if len(redshifts) > 0:
        print("\nd_A at z=" + str(redshifts[0]) + ": " + str(angular_distances[0]) + " Mpc")
    if len(redshifts) > 10:  # e.g. z around 0.4
        idx_z_0_4 = np.abs(redshifts - 0.4).argmin()
        print("d_A at z=" + str(redshifts[idx_z_0_4]) + ": " + str(angular_distances[idx_z_0_4]) + " Mpc")
    if len(redshifts) > 25:  # e.g. z around 1.0
        idx_z_1_0 = np.abs(redshifts - 1.0).argmin()
        print("d_A at z=" + str(redshifts[idx_z_1_0]) + ": " + str(angular_distances[idx_z_1_0]) + " Mpc")
    if len(redshifts) > 0:
        print("d_A at z=" + str(redshifts[-1]) + ": " + str(angular_distances[-1]) + " Mpc")


if __name__ == '__main__':
    calculate_angular_diameter_distance()