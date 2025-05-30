# filename: codebase/calculate_cmb_power_spectrum.py
import os
import numpy as np
import camb

def calculate_cmb_power_spectrum():
    """
    Calculates the CMB raw temperature power spectrum using CAMB for a specific
    Lambda CDM cosmology and saves the results to a CSV file.

    The cosmological parameters are:
    - Hubble constant (H0): 74 km/s/Mpc
    - Baryon density (ombh2): 0.022
    - Cold dark matter density (omch2): 0.122
    - Neutrino mass sum (mnu): 0.06 eV
    - Curvature (omk): 0
    - Optical depth to reionization (tau): 0.06
    - Scalar amplitude (As): 2e-9
    - Scalar spectral index (ns): 0.965

    The temperature power spectrum (C_l^TT) is computed in units of muK^2
    for multipole moments l = 2 to l = 3000.
    The results are saved in 'data/result.csv'.
    """
    print("Starting CMB power spectrum calculation...")

    # Cosmological parameters
    H0_val = 74.0  # Hubble constant in km/s/Mpc
    ombh2_val = 0.022  # Physical baryon density
    omch2_val = 0.122  # Physical cold dark matter density
    mnu_val = 0.06  # Sum of neutrino masses in eV
    omk_val = 0.0  # Curvature parameter
    tau_val = 0.06  # Optical depth to reionization
    As_val = 2.0e-9  # Scalar amplitude
    ns_val = 0.965  # Scalar spectral index
    lmax_val = 3000  # Maximum multipole moment

    try:
        # Set up CAMB parameters
        pars = camb.CAMBparams()
        print("Setting cosmological parameters...")
        pars.set_cosmology(H0=H0_val, ombh2=ombh2_val, omch2=omch2_val, mnu=mnu_val, omk=omk_val, tau=tau_val)
        
        print("Setting initial power spectrum parameters...")
        pars.set_initial_power(As=As_val, ns=ns_val)
        
        print("Setting lmax and accuracy for unlensed Cls...")
        # lens_potential_accuracy=0 for unlensed Cls, which "raw" implies
        pars.set_for_lmax(lmax=lmax_val, lens_potential_accuracy=0)

        # Run CAMB
        print("Running CAMB to get results...")
        results = camb.get_results(pars)

        # Get unlensed scalar Cls
        # CMB_unit='muK' returns C_l in muK^2
        print("Extracting unlensed scalar power spectra (C_l^TT) in muK^2...")
        # cls is an array of shape (nspectra, lmax+1)
        # For unlensed scalar, nspectra=4 (TT, EE, BB, TE)
        # cls[0,:] is TT, cls[1,:] is EE, etc.
        # Values are C_l, from l=0 to lmax.
        cls = results.get_unlensed_scalar_cls(lmax=lmax_val, CMB_unit='muK')

        # Extract l and TT spectrum for l=2 to lmax_val
        # CAMB output cls[0,l] corresponds to multipole l.
        # We need l from 2 to lmax_val.
        ls = np.arange(2, lmax_val + 1)
        # cls[0,:] contains C_l^TT for l=0, 1, ..., lmax_val
        # We need to slice this from index 2 up to index lmax_val (inclusive)
        cl_TT = cls[0, 2 : lmax_val + 1]  # Units: muK^2

        # Ensure the lengths match
        if len(ls) != len(cl_TT):
            raise ValueError("Mismatch in length of l and C_l^TT arrays. This should not happen.")

        # Prepare data for CSV
        output_data = np.column_stack((ls, cl_TT))

        # Create data directory if it doesn't exist
        output_dir = "data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print("Created directory: " + output_dir)

        # Save to CSV
        output_filename = os.path.join(output_dir, "result.csv")
        print("Saving results to " + output_filename + "...")
        # Header: l,TT. No comments. Delimiter is comma.
        np.savetxt(output_filename, output_data, delimiter=',', header='l,TT', comments='', fmt=['%d', '%.18e'])
        
        print("CMB power spectrum calculation and data export completed successfully.")
        print("Output file: " + output_filename)
        print("Columns: l (multipole moment), TT (raw temperature power spectrum in muK^2)")
        print("Number of data points: " + str(len(ls)))
        print("l range: " + str(ls[0]) + " to " + str(ls[-1]))
        
        # Print some sample values
        if len(ls) > 5:
            print("Sample data (first 5 rows):")
            for i in range(min(5, len(ls))):
                print("l=" + str(int(output_data[i,0])) + ", TT=" + str(output_data[i,1]))
        else:
            print("Sample data:")
            for i in range(len(ls)):
                 print("l=" + str(int(output_data[i,0])) + ", TT=" + str(output_data[i,1]))


    except Exception as e:
        print("An error occurred during the CAMB calculation or data export:")
        print(str(e))


if __name__ == '__main__':
    calculate_cmb_power_spectrum()
    # Print a completion status for the overall script execution for the orchestrator
    # This is not part of the CAMB calculation itself but for workflow management.
    # print("##SCRIPT_COMPLETED_SUCCESSFULLY##") # This line is for automated systems, not for direct output.
