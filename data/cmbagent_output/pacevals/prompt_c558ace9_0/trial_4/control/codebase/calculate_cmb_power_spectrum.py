# filename: codebase/calculate_cmb_power_spectrum.py
import camb
import numpy as np
import os


def calculate_cmb_power_spectrum():
    """
    Calculates the CMB raw temperature power spectrum for a flat Lambda CDM cosmology
    using CAMB and saves the results to a CSV file.

    The cosmological parameters used are:
    H_0 = 74 km/s/Mpc
    Omega_b h^2 = 0.022
    Omega_c h^2 = 0.122
    Sum of neutrino masses = 0.06 eV
    Omega_k = 0
    tau = 0.06
    A_s = 2e-9
    n_s = 0.965

    The temperature power spectrum (C_l^TT) is computed in units of muK^2
    for multipole moments l = 2 to l = 3000.
    The results are saved in 'data/result.csv'.
    """
    print("Starting CMB power spectrum calculation with CAMB...")

    # Define cosmological parameters
    H0 = 74.0  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Physical baryon density Omega_b h^2
    omch2 = 0.122  # Physical cold dark matter density Omega_c h^2
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature density Omega_k
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar power spectrum amplitude
    ns = 0.965  # Scalar spectral index
    lmax = 3000  # Maximum multipole moment for calculation

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.set_initial_power(As=As, ns=ns)

    # Set lmax and specify we want unlensed spectra, so lensing calculations can be skipped/simplified
    # lens_potential_accuracy=0 means no lensing calculation for the potential itself.
    pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=0)

    # We want unlensed scalar spectra.
    # DoLensing=False ensures CAMB doesn't compute lensed spectra or lensing potential.
    # Default for DoLensing is True.
    pars.WantScalars = True
    pars.DoLensing = False

    # Get results
    print("Running CAMB... This might take a few moments.")
    results = camb.get_results(pars)
    print("CAMB run completed.")

    # Get unlensed scalar Cls (TT, EE, BB, TE)
    # These are returned in units of muK^2
    # The array 'powers' will have shape (lmax+1, 4)
    # Column 0: TT, Column 1: EE, Column 2: BB, Column 3: TE
    powers = results.get_unlensed_scalar_cls(lmax=lmax)

    # Extract TT spectrum for l=2 to lmax
    # ls is an array of multipole moments from 2 to lmax
    ls = np.arange(2, lmax + 1)
    # cl_TT contains the C_l^TT values corresponding to ls.
    # powers[l,0] gives C_l^TT. We need from l=2.
    cl_TT = powers[2:lmax + 1, 0]  # Units: muK^2

    # Prepare data for saving: a 2D array with l and C_l^TT
    output_data = np.column_stack((ls, cl_TT))

    # Define output directory and file path
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: " + output_dir)
    
    file_path = os.path.join(output_dir, "result.csv")

    # Save to CSV file
    # Header: "l,TT"
    # Delimiter: ","
    # Comments: No comment character (comments='')
    # Format: l as integer ('%d'), TT as float with 18 decimal places in scientific notation ('%.18e')
    print("Saving results to " + file_path + "...")
    np.savetxt(file_path, output_data, header="l,TT", delimiter=",", comments="", fmt=['%d', '%.18e'])
    print("Results saved successfully to " + file_path)

    # Print some of the results to console for verification
    print("\nSample of the calculated power spectrum (l,TT value):")
    print("First 5 rows:")
    for i in range(min(5, len(output_data))):
        # Print l (integer) and TT value (float)
        print(str(int(output_data[i,0])) + "," + str(output_data[i,1]))

    print("\nLast 5 rows:")
    # Ensure we don't try to print more rows than exist if lmax is small
    start_index_last_rows = max(0, len(output_data)-5)
    for i in range(start_index_last_rows, len(output_data)):
        # Print l (integer) and TT value (float)
        print(str(int(output_data[i,0])) + "," + str(output_data[i,1]))

    print("\nCMB power spectrum calculation and saving finished.")


if __name__ == '__main__':
    calculate_cmb_power_spectrum()