# filename: codebase/cmb_b_mode.py
import os
import numpy as np
import pandas as pd
import camb

def calculate_cmb_b_mode_power_spectrum():
    r"""Calculates the CMB B-mode polarization power spectrum l(l+1)C_l^BB/(2pi)
    for a flat Lambda CDM cosmology using CAMB.

    The calculation uses the following cosmological parameters:
    - Hubble constant (H0): 67.5 km/s/Mpc
    - Baryon density (ombh2): 0.022
    - Cold dark matter density (omch2): 0.122
    - Neutrino mass sum (mnu): 0.06 eV
    - Curvature (omk): 0
    - Optical depth to reionization (tau): 0.06
    - Tensor-to-scalar ratio (r): 0
    - Scalar amplitude (As): 2e-9
    - Scalar spectral index (ns): 0.965

    The B-mode power spectrum is computed in units of muK^2 for multipole
    moments l from 2 to 3000. The results are saved in a CSV file.

    Returns:
        str: The file path of the saved CSV file.
    """
    # Define cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Physical baryon density
    omch2 = 0.122  # Physical cold dark matter density
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature density
    tau = 0.06  # Optical depth to reionization

    # Initial power spectrum parameters
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    r_tensor_to_scalar = 0.0  # Tensor-to-scalar ratio

    # Maximum multipole for output
    lmax_output = 3000

    # Initialize CAMB parameters object
    pars = camb.CAMBparams()

    # Set cosmology
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)

    # Set initial power spectrum parameters
    pars.InitPower.set_params(As=As, ns=ns, r=r_tensor_to_scalar)

    # Set lmax for calculation (slightly higher than output lmax for accuracy)
    # and lensing accuracy.
    pars.set_for_lmax(lmax_output + 100, lens_potential_accuracy=1)
    
    # Since r=0, primordial tensor modes are zero.
    # We are interested in lensed scalar B-modes.
    # The default pars.WantTensors = False is appropriate.

    # Get results from CAMB
    results = camb.get_results(pars)

    # Get lensed scalar power spectra C_l.
    # CMB_unit='muK' ensures C_l are in muK^2. This is often the default.
    powers = results.get_lensed_scalar_cls(lmax=lmax_output, CMB_unit='muK')

    # Extract l values and C_l^BB
    # powers['BB'] is an array of C_l^BB values, indexed from l=0 to lmax_output.
    # Its length is lmax_output + 1.
    ls = np.arange(powers['BB'].size)  # Multipole moments l, type int
    cl_bb = powers['BB']  # C_l^BB in muK^2, type float

    # Filter for l in the range [2, lmax_output]
    mask = (ls >= 2) & (ls <= lmax_output)
    ls_filtered = ls[mask]
    cl_bb_filtered = cl_bb[mask]  # C_l^BB in muK^2 for l in [2, lmax_output]

    # Calculate l(l+1)C_l^BB / (2pi)
    dl_bb = ls_filtered * (ls_filtered + 1) * cl_bb_filtered / (2 * np.pi)  # muK^2

    # Create a pandas DataFrame
    df = pd.DataFrame({'l': ls_filtered, 'BB': dl_bb})
    df['l'] = df['l'].astype(int)  # Ensure 'l' column is integer

    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save to CSV
    file_path = os.path.join(data_dir, "result.csv")
    df.to_csv(file_path, index=False)

    # Print information about the saved data
    print("CMB B-mode power spectrum calculation complete.")
    print("Data saved to: " + file_path)
    print("\nFirst 5 rows of the B-mode power spectrum data (l, l(l+1)C_l^BB/(2pi) in muK^2):")
    print(df.head().to_string())
    print("\nLast 5 rows of the B-mode power spectrum data (l, l(l+1)C_l^BB/(2pi) in muK^2):")
    print(df.tail().to_string())
    
    # Create a buffer for DataFrame info
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    print("\nDataFrame info:")
    print(info_str)

    return file_path


if __name__ == '__main__':
    # Execute the calculation and save results
    # This block will run when the script is executed directly.
    # Suppress CAMB's own print statements about versions etc.
    # Note: CAMB may still print some output depending on its internal configuration.
    # For full silence, one might need to redirect stdout/stderr for the CAMB calls,
    # but this is generally not needed unless output is excessive.
    
    # Set pandas display options for better console output
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 5)
    pd.set_option('display.width', 120)

    saved_file_path = calculate_cmb_b_mode_power_spectrum()
    # The function already prints the path, so this is mostly for programmatic use if needed.
    # print("Process finished. Results are in " + saved_file_path)