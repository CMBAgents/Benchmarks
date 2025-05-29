# filename: codebase/delensed_cmb_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def calculate_delensed_cmb_spectrum():
    r"""
    Calculates the delensed CMB temperature power spectrum and saves it to a CSV file.

    The function performs the following steps:
    1. Sets up cosmological parameters for CAMB based on the problem specification.
    2. Computes lensed and unlensed scalar TT power spectra using CAMB.
       The spectra are D_l = l(l+1)C_l^TT / (2pi) in muK^2.
    3. Applies an 80% delensing efficiency to calculate the delensed TT spectrum.
    4. Extracts the spectrum for multipole moments l from 2 to 3000.
    5. Saves the results ('l' and 'TT' columns) to 'data/result.csv'.
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Baryon density Omega_b * h^2
    omch2 = 0.122  # Cold dark matter density Omega_c * h^2
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature Omega_k
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    
    delensing_efficiency = 0.8 # 80%

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    
    # Set lmax for CAMB internal calculations and specify lensed Cl calculation
    # Using a higher lmax for internal calculations (e.g., 4000) for accuracy up to l_output=3000
    pars.set_for_lmax(lmax=4000, lens_potential_accuracy=1)
    pars.DoLensing = True # Ensure lensed spectra are calculated by CAMB

    # Get results from CAMB
    results = camb.get_results(pars)

    # Get power spectra D_l = l(l+1)C_l/(2pi) in muK^2
    # We need spectra up to l=3000 for the output.
    lmax_output = 3000
    powers = results.get_cmb_power_spectra(pars, lmax=lmax_output, 
                                           spectra=['total', 'unlensed_scalar'], CMB_unit='muK')

    # powers['total'] contains lensed D_l (TT, EE, BB, TE)
    # powers['unlensed_scalar'] contains unlensed scalar D_l (TT, EE, BB, TE)
    # Each is a (lmax_output+1, 4) array, for l from 0 to lmax_output.
    # We need the TT component, which is the first column (index 0).
    Dl_lensed_TT = powers['total'][:, 0]
    Dl_unlensed_TT = powers['unlensed_scalar'][:, 0]

    # Multipole moments 'l' for the output range (2 to 3000)
    # np.arange(lmax_output + 1) gives l from 0 to 3000.
    ls_all = np.arange(lmax_output + 1) 
    
    # Select l values from 2 to 3000
    # The spectra arrays are 0-indexed, so l=2 corresponds to index 2.
    l_values_for_csv = ls_all[2:] 
    Dl_lensed_TT_slice = Dl_lensed_TT[2:]
    Dl_unlensed_TT_slice = Dl_unlensed_TT[2:]

    # Calculate the delensed TT power spectrum
    # D_l_delensed = D_l_unlensed + (1 - efficiency) * (D_l_lensed - D_l_unlensed)
    Dl_delensed_TT_final = Dl_unlensed_TT_slice + \
                           (1.0 - delensing_efficiency) * (Dl_lensed_TT_slice - Dl_unlensed_TT_slice)

    # Prepare DataFrame for CSV export
    df = pd.DataFrame({
        'l': l_values_for_csv.astype(int), # Ensure 'l' is integer
        'TT': Dl_delensed_TT_final  # Delensed D_l^TT in muK^2
    })

    # Create data directory if it doesn't exist
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    csv_filename = "result.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path, index=False)

    print("Successfully calculated delensed CMB temperature power spectrum.")
    print("Results saved to: " + csv_path)
    print("The CSV file has two columns: 'l' (multipole moment, " + str(df['l'].min()) + " to " + str(df['l'].max()) + ") and 'TT' (delensed D_l^TT = l(l+1)C_l^TT/(2pi) in muK^2).")
    print("Number of rows in CSV: " + str(len(df)))
    
    # Print some sample data from the DataFrame
    # Using to_string() to ensure full display for verification, rather than truncated repr
    if len(df) > 5:
        print("First 5 rows of the data:")
        print(df.head().to_string())
    else:
        print("Data:")
        print(df.to_string())


if __name__ == '__main__':
    calculate_delensed_cmb_spectrum()