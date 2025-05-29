# filename: codebase/calculate_angular_diameter_distance.py
import camb
import numpy as np
import pandas as pd
import os

def calculate_angular_diameter_distance():
    r"""
    Calculates the angular diameter distance (d_A) for a range of redshifts
    using specified cosmological parameters with CAMB.

    The cosmological parameters are:
    - Hubble constant (H0): 67.5 km/s/Mpc
    - Baryon density (ombh2): 0.022
    - Cold dark matter density (omch2): 0.122
    - Neutrino mass sum (mnu): 0.06 eV
    - Curvature (omk): 0
    - Optical depth to reionization (tau): 0.06
    - Scalar amplitude (As): 2e-9
    - Scalar spectral index (ns): 0.965

    The angular diameter distance is computed for 100 evenly spaced redshift
    points from z=0 to z=4. The results are saved in a CSV file named
    'result.csv' in the 'data/' directory, with columns 'z' and 'd_A' (in Mpc).
    The results are also printed to the console.
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Physical baryon density
    omch2 = 0.122  # Physical cold dark matter density
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature density
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    
    # Redshift range
    redshifts = np.linspace(0, 4, 100)  # 100 evenly spaced points from z=0 to z=4

    # Calculate angular diameter distances
    # We need to get results from CAMB.
    # To get background results, we can call get_background.
    # However, angular_diameter_distance is a method of the CAMBdata object,
    # so we need to run calculations first.
    pars.set_for_lmax(2500, lens_potential_accuracy=0)  # Basic settings for background
    results = camb.get_results(pars)
    
    # Angular diameter distance in Mpc
    # The function angular_diameter_distance can take an array of redshifts
    # For z=0, d_A is 0. CAMB might return a very small number or handle it.
    # Let's check CAMB's behavior for z=0.
    # angular_diameter_distance(0) should be 0.
    # If redshifts[0] is 0, results.angular_diameter_distance(redshifts[0]) might cause issues
    # if CAMB's internal calculations don't handle z=0 gracefully for d_A.
    # However, typically d_A(0) = 0 is well-defined.
    
    # CAMB's angular_diameter_distance function can take a scalar or an array.
    # It returns d_A in Mpc.
    angular_distances = results.angular_diameter_distance(redshifts)  # d_A in Mpc

    # Create a Pandas DataFrame
    df = pd.DataFrame({'z': redshifts, 'd_A': angular_distances})

    # Create data directory if it doesn't exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save to CSV
    file_path = os.path.join(data_dir, 'result.csv')
    df.to_csv(file_path, index=False)
    print("Angular diameter distances saved to: " + str(file_path))

    # Print results to console
    print("\nCalculated Angular Diameter Distances (Mpc):")
    # Set pandas display options to show all rows and more precision
    pd.set_option('display.max_rows', None)
    pd.set_option('display.precision', 5)
    print(df)


if __name__ == '__main__':
    calculate_angular_diameter_distance()