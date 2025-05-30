# filename: codebase/calculate_cmb_power_spectrum.py
import os
import camb
import numpy as np
import pandas as pd

def calculate_cmb_power_spectrum():
    """
    Calculates the CMB raw temperature power spectrum (C_l^TT) for a flat Lambda CDM cosmology
    using specified parameters with CAMB. Saves the results to a CSV file.
    """
    # Cosmological parameters
    H0 = 74.0  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Physical baryon density
    omch2 = 0.122  # Physical cold dark matter density
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature density
    tau = 0.06  # Optical depth to reionization
    As = 2.0e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    lmax = 3000  # Maximum multipole moment

    # Output directory and file
    output_dir = "data"
    output_filename = "result.csv"
    output_path = os.path.join(output_dir, output_filename)

    # Create output directory if it doesn't exist
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print("Created directory: " + output_dir)
    except OSError as e:
        print("Error creating directory " + output_dir + ": " + str(e))
        return

    try:
        # Initialize CAMB parameters
        pars = camb.CAMBparams()

        # Set cosmological parameters
        # H0: Hubble constant (km/s/Mpc)
        # ombh2: Physical baryon density Omega_b*h^2
        # omch2: Physical cold dark matter density Omega_c*h^2
        # omk: Curvature Omega_k
        # tau: Optical depth to reionization
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omk, tau=tau)

        # Set massive neutrino parameters
        # mnu: Sum of neutrino masses (eV)
        # num_massive_neutrinos: Number of massive neutrino species.
        # We assume 3 degenerate massive neutrinos, so CAMB will distribute mnu among them.
        pars.set_massive_neutrinos(mnu=mnu, num_massive_neutrinos=3)

        # Set initial power spectrum parameters
        # As: Scalar fluctuation amplitude
        # ns: Scalar spectral index
        pars.InitPower.set_params(As=As, ns=ns)

        # Set calculation range and accuracy for lensed spectra
        # lmax: Maximum multipole l
        # lens_potential_accuracy: Controls accuracy of lensing potential calculation (1 is good for typical use)
        pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=1)

        print("CAMB parameters set. Starting calculations...")
        # Get results from CAMB
        results = camb.get_results(pars)
        print("CAMB calculations finished.")

        # Get CMB power spectra
        # CMB_unit='muK': Output C_l in muK^2
        # raw_cl=True: Output C_l (not l(l+1)C_l/2pi)
        # spectra=['lensed_scalar']: Compute lensed scalar power spectra
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True, spectra=['lensed_scalar'])
        
        # powers is a dictionary. For lensed_scalar, it returns an array with columns TT, EE, BB, TE
        # We need the TT spectrum (temperature-temperature), which is the first column (index 0)
        # The array is (lmax+1) x 4. Index l corresponds to multipole l.
        cl_tt_all = powers['lensed_scalar'][:, 0]  # Units: muK^2

        # We need C_l^TT for l from 2 to lmax (inclusive)
        l_values = np.arange(2, lmax + 1).astype(int)
        cl_tt_segment = cl_tt_all[2:lmax + 1] # cl_tt_all[l] is C_l

        # Create a Pandas DataFrame
        results_df = pd.DataFrame({
            'l': l_values,      # Multipole moment
            'TT': cl_tt_segment # Raw temperature power spectrum (muK^2)
        })

        # Save to CSV
        results_df.to_csv(output_path, index=False)
        print("Successfully saved CMB power spectrum to: " + output_path)
        
        # Print head of the DataFrame to console
        print("\nFirst 5 rows of the results:")
        # Using to_string() for better console output of pandas DataFrame head
        print(results_df.head().to_string())
        
        # Print some summary statistics
        print("\nSummary of calculated TT power spectrum:")
        print("Number of l values: " + str(len(results_df)))
        print("Min l: " + str(results_df['l'].min()))
        print("Max l: " + str(results_df['l'].max()))
        print("Mean TT value (muK^2): " + str(results_df['TT'].mean()))


    except Exception as e:
        print("An error occurred during CAMB calculation or data processing:")
        import traceback
        print(traceback.format_exc())


if __name__ == '__main__':
    calculate_cmb_power_spectrum()