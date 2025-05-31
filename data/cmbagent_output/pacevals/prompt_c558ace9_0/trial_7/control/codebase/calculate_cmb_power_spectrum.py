# filename: codebase/calculate_cmb_power_spectrum.py
import camb
import numpy as np
import pandas as pd
import os


def calculate_cmb_power_spectrum():
    """
    Calculates the CMB raw temperature power spectrum (C_l^TT) for a flat Lambda CDM cosmology
    using specified parameters with CAMB. Saves the results to a CSV file.

    Cosmological Parameters:
        Hubble constant (H0): 74 km/s/Mpc
        Baryon density (ombh2): 0.022
        Cold dark matter density (omch2): 0.122
        Neutrino mass sum (mnu): 0.06 eV
        Curvature (omk): 0
        Optical depth to reionization (tau): 0.06
        Scalar amplitude (As): 2e-9
        Scalar spectral index (ns): 0.965
    
    Output:
        Saves a CSV file 'data/result.csv' with columns 'l' and 'TT'.
        'l': Multipole moment (integer values from 2 to 3000)
        'TT': Raw temperature power spectrum (C_l^TT) in muK^2
    """
    try:
        # Define cosmological parameters
        H0_val = 74.0  # Hubble constant in km/s/Mpc
        ombh2_val = 0.022  # Physical baryon density
        omch2_val = 0.122  # Physical cold dark matter density
        mnu_val = 0.06  # Sum of neutrino masses in eV
        omk_val = 0.0  # Curvature density
        tau_val = 0.06  # Optical depth to reionization
        As_val = 2.0e-9  # Scalar amplitude
        ns_val = 0.965  # Scalar spectral index
        lmax_val = 3000 # Maximum multipole moment

        # Set up CAMB parameters
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0_val, ombh2=ombh2_val, omch2=omch2_val, mnu=mnu_val, omk=omk_val, tau=tau_val)
        pars.InitPower.set_params(As=As_val, ns=ns_val)
        
        # We want unlensed scalar spectra.
        # lens_potential_accuracy=0 would disable lensing calculations entirely.
        # lens_potential_accuracy=1 is standard if lensed spectra or lensing potential are also needed.
        # Since we only extract 'unlensed_scalar', CAMB will provide it.
        # For efficiency, if only unlensed is ever needed, lens_potential_accuracy=0 could be used.
        # However, using 1 and extracting unlensed is also fine and common.
        pars.set_for_lmax(lmax_val, lens_potential_accuracy=1)

        # Specify that we want scalar modes (for C_l^TT)
        pars.WantScalars = True
        pars.WantTensors = False  # Not requested
        pars.WantVectors = False  # Not requested

        # Run CAMB calculation
        print("Running CAMB calculation...")
        results = camb.get_results(pars)
        print("CAMB calculation finished.")

        # Get raw (unlensed) C_l^TT in muK^2
        # 'unlensed_scalar': TT, EE, BB, TE
        # We need the first column (TT)
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', spectra=['unlensed_scalar'])
        cl_tt_unlensed = powers['unlensed_scalar'][:, 0]  # C_l^TT in muK^2

        # Prepare l and TT arrays for l=2 to l=3000
        # CAMB output array is 0-indexed by l. So, cl_tt_unlensed[l] corresponds to C_l.
        # We need l from 2 up to lmax_val (inclusive).
        l_values = np.arange(2, lmax_val + 1)
        
        # Check if cl_tt_unlensed has enough elements
        if len(cl_tt_unlensed) <= lmax_val:
            # This might happen if lmax_val was too high for some internal CAMB limit or setting
            # Or if the array is shorter than expected for other reasons.
            # Adjust l_values to match available data if necessary, or raise an error.
            # For this problem, CAMB should produce up to lmax_val.
            actual_lmax_computed = len(cl_tt_unlensed) - 1
            if actual_lmax_computed < lmax_val:
                print("Warning: CAMB computed up to l=" + str(actual_lmax_computed) + ", which is less than requested lmax=" + str(lmax_val) + ".")
                print("Proceeding with available data up to l=" + str(actual_lmax_computed) + ".")
                if actual_lmax_computed < 2:
                    raise ValueError("CAMB did not compute C_l for l >= 2.")
                l_values = np.arange(2, actual_lmax_computed + 1)
                cl_tt_values = cl_tt_unlensed[2 : actual_lmax_computed + 1]
            else:  # len(cl_tt_unlensed) > lmax_val (e.g. lmax_val + 1 elements if 0-indexed)
                cl_tt_values = cl_tt_unlensed[2 : lmax_val + 1]
        else:  # len(cl_tt_unlensed) > lmax_val + 1, which is expected
            cl_tt_values = cl_tt_unlensed[2 : lmax_val + 1]

        # Create pandas DataFrame
        df = pd.DataFrame({"l": l_values, "TT": cl_tt_values})

        # Define output directory and filename
        output_dir = 'data'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        file_path = os.path.join(output_dir, 'result.csv')

        # Save to CSV
        df.to_csv(file_path, index=False)
        print("CMB power spectrum data saved to: " + str(file_path))
        
        # Print some info about the generated data
        print("\nFirst 5 rows of the generated data (result.csv):")
        print(df.head().to_string())
        
        print("\nSummary of TT values:")
        print("Min TT: " + str(df['TT'].min()))
        print("Max TT: " + str(df['TT'].max()))
        print("Mean TT: " + str(df['TT'].mean()))
        print("Number of l values: " + str(len(df)))

    except Exception as e:
        print("An error occurred:")
        import traceback
        print(traceback.format_exc())


if __name__ == '__main__':
    calculate_cmb_power_spectrum()
    # Record script completion status
    # This is a placeholder for the actual status recording mechanism
    # if 'default_api' in globals() and hasattr(default_api, 'record_status_SUCCESS'):
    #    print(default_api.record_status_SUCCESS())
    # else:
    #    print("Status recording API not available.")
    # For the purpose of this environment, we'll just print a success message.
    print("\nScript finished.")
