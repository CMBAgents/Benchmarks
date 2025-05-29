# filename: codebase/calculate_angular_diameter_distance.py
import camb
import numpy as np
import pandas as pd
import os

def calculate_angular_diameter_distance():
    r"""
    Calculates the angular diameter distance (d_A) for a range of redshifts
    using specified cosmological parameters with CAMB.

    The function sets up a flat Lambda CDM cosmology, computes d_A for 100
    redshift points between z=0 and z=4, and saves the results to a CSV file.

    Cosmological Parameters:
        Hubble constant (H0): 67.5 km/s/Mpc
        Baryon density (ombh2): 0.022
        Cold dark matter density (omch2): 0.122
        Neutrino mass sum (mnu): 0.06 eV
        Curvature (omk): 0
        Optical depth to reionization (tau): 0.06
        Scalar amplitude (As): 2e-9
        Scalar spectral index (ns): 0.965
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
    pars.set_for_background_observations(lens_potential_accuracy=1)  # lens_potential_accuracy is not strictly needed for d_A but good practice

    # Redshift range
    redshifts = np.linspace(0, 4, 100)  # 100 evenly spaced redshift points

    # Calculate angular diameter distances
    # CAMB's angular_diameter_distance function returns an array if redshifts is an array.
    # The result is in Mpc.
    try:
        # For z=0, d_A is 0. CAMB might return a very small number or handle it internally.
        # We can compute for z > 0 and handle z=0 separately if needed,
        # but CAMB's angular_diameter_distance function should handle an array of redshifts including 0.
        # However, results.angular_diameter_distance(0) might raise an error or return NaN
        # depending on the CAMB version or specific settings if not careful.
        # It's safer to calculate for z > 0 and then prepend the (0,0) point if issues arise.
        # Let's test with the full array first.
        
        # Get background cosmology results
        results = camb.get_background(pars)
        
        # Calculate angular diameter distances for the array of redshifts
        # angular_diameter_distance(z) returns d_A in Mpc
        # For an array of redshifts, it returns an array of d_A values
        d_A_values = results.angular_diameter_distance(redshifts)  # d_A in Mpc

    except Exception as e:
        print("Error during CAMB calculation: " + str(e))
        # Fallback or error handling if CAMB fails for z=0 or other reasons
        # For this specific task, we expect CAMB to handle it.
        # If not, one might compute for redshifts[1:] and prepend (0,0)
        # For now, let's assume it works.
        return

    # Create a Pandas DataFrame
    df = pd.DataFrame({'z': redshifts, 'd_A': d_A_values})
    df['d_A'] = df['d_A'].astype(float)  # Ensure d_A is float

    # Create data directory if it doesn't exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save to CSV
    file_path = os.path.join(data_dir, 'result.csv')
    df.to_csv(file_path, index=False)
    print("Results saved to " + str(file_path))

    # Print the DataFrame to console
    print("\nAngular Diameter Distances:")
    # Set pandas display options to show all rows and ensure float precision
    pd.set_option('display.max_rows', None)
    pd.set_option('display.precision', 6)
    print(df)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.precision')


if __name__ == '__main__':
    calculate_angular_diameter_distance()