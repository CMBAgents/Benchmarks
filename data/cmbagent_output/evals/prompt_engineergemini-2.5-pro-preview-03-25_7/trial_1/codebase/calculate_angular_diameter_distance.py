# filename: codebase/calculate_angular_diameter_distance.py
import numpy as np
import pandas as pd
import camb
import os

def calculate_angular_diameter_distance():
    r"""
    Calculates the angular diameter distance (d_A) for a flat Lambda CDM cosmology
    using specified parameters with CAMB.

    The calculation is performed for 100 evenly spaced redshift points from z=0 to z=4.
    The results (redshift and d_A in Mpc) are saved to a CSV file.
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Baryon density
    omch2 = 0.122  # Cold dark matter density
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.WantCls = False # We only need background cosmology, not C_ls

    # Redshift range
    z_values = np.linspace(0, 4, 100) # Redshift (dimensionless)

    # Calculate angular diameter distances
    # CAMB's angular_diameter_distance function takes an array of redshifts
    # and returns an array of d_A in Mpc.
    # We need to get the background results first for the specified redshifts.
    
    # To get results at specific redshifts, we can set them directly
    # However, angular_diameter_distance is a method of the results object from get_background
    # which itself can take a redshift array.
    # For efficiency, it's better to compute all at once if possible.
    # CAMB's `get_background` can compute results up to a max redshift,
    # and then `angular_diameter_distance` can be called with an array of redshifts.

    pars.set_matter_power(redshifts=z_values.tolist(), kmax=2.0) # Ensure CAMB computes up to max z
    
    results = camb.get_background(pars)
    # angular_diameter_distance(z) returns d_A in Mpc
    # For z=0, d_A is expected to be 0. CAMB might return a very small number or handle it.
    # Let's check CAMB's behavior for z=0.
    # The function angular_diameter_distance can take a scalar or an array.
    
    # Calculate d_A for each z in z_values
    # The angular_diameter_distance function in CAMB results object can take an array of redshifts.
    da_values = results.angular_diameter_distance(z_values) # d_A in Mpc

    # Create a Pandas DataFrame
    df_results = pd.DataFrame({'z': z_values, 'd_A': da_values})

    # Ensure the data directory exists
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save results to CSV
    file_path = os.path.join(data_dir, 'result.csv')
    df_results.to_csv(file_path, index=False)

    print("Angular diameter distance calculation complete.")
    print("Results saved to: " + str(file_path))
    print("\nFirst 5 rows of the results:")
    # Configure pandas to display float with more precision if needed
    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    print(df_results.head())
    
    # Print some specific values for verification
    if len(z_values) > 0:
        print("\nd_A at z=" + str(z_values[0]) + ": " + str(da_values[0]) + " Mpc")
    if len(z_values) > 1:
        idx_z1 = np.where(np.isclose(z_values, 1.0))[0]
        if len(idx_z1) > 0:
             print("d_A at z~1.0 (z=" + str(z_values[idx_z1[0]]) + "): " + str(da_values[idx_z1[0]]) + " Mpc")
    if len(z_values) > 0:
        print("d_A at z=" + str(z_values[-1]) + ": " + str(da_values[-1]) + " Mpc")


if __name__ == '__main__':
    calculate_angular_diameter_distance()