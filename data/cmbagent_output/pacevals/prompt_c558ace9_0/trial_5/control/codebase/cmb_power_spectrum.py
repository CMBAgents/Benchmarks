# filename: codebase/cmb_power_spectrum.py
import camb
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import matplotlib

matplotlib.rcParams['text.usetex'] = False # Disable LaTeX rendering

def calculate_cmb_power_spectrum():
    """
    Calculates the CMB raw temperature power spectrum (C_l^TT) for a flat Lambda CDM cosmology
    using specified parameters with CAMB.

    The function sets up cosmological parameters, runs CAMB, and extracts the
    temperature power spectrum C_l^TT in units of muK^2 for multipole moments
    from l=2 to l=3000.

    Cosmological Parameters:
        Hubble constant (H0): 74 km/s/Mpc
        Baryon density (ombh2): 0.022
        Cold dark matter density (omch2): 0.122
        Neutrino mass sum (mnu): 0.06 eV
        Curvature (omk): 0
        Optical depth to reionization (tau): 0.06
        Scalar amplitude (As): 2e-9
        Scalar spectral index (ns): 0.965
        Maximum multipole (lmax): 3000

    Returns:
        tuple: A tuple containing:
            - ls (numpy.ndarray): Array of multipole moments from 2 to 3000.
            - cl_TT (numpy.ndarray): Array of raw temperature power spectrum C_l^TT in muK^2.
            Returns (None, None) if an error occurs.
    """
    try:
        # 1. Set up CAMB parameters
        pars = camb.CAMBparams()

        # Cosmological parameters
        h0_val = 74.0  # Hubble constant in km/s/Mpc
        ombh2_val = 0.022  # Physical baryon density
        omch2_val = 0.122  # Physical cold dark matter density
        mnu_val = 0.06  # Sum of neutrino masses in eV
        omk_val = 0.0  # Curvature
        tau_val = 0.06  # Optical depth to reionization

        pars.set_cosmology(H0=h0_val, ombh2=ombh2_val, omch2=omch2_val, mnu=mnu_val, omk=omk_val)
        pars.Reion.optical_depth = tau_val

        
        # Initial power spectrum parameters
        as_val = 2e-9  # Scalar amplitude
        ns_val = 0.965  # Scalar spectral index
        pars.InitPower.set_params(As=as_val, ns=ns_val)

        # Set maximum multipole and lensing
        lmax_val = 3000  # Maximum multipole moment
        pars.set_for_lmax(lmax_val, lens_potential_accuracy=1)

        print("CAMB parameters set:")
        print("H0: " + str(h0_val) + " km/s/Mpc")
        print("ombh2: " + str(ombh2_val))
        print("omch2: " + str(omch2_val))
        print("mnu: " + str(mnu_val) + " eV")
        print("omk: " + str(omk_val))
        print("tau: " + str(tau_val))
        print("As: " + str(as_val))
        print("ns: " + str(ns_val))
        print("lmax: " + str(lmax_val))

        # 2. Get results
        print("Running CAMB to get results...")
        results = camb.get_results(pars)
        print("CAMB calculation finished.")

        # 3. Get CMB power spectra
        print("Extracting CMB power spectra...")
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True, lensed_cl=True)
        
        all_cls = powers['total']
        if all_cls.shape[1] < 1:
            print("Error: 'total' spectrum does not contain TT data or is empty.")
            return None, None
            
        print("Shape of all_cls ('total'): " + str(all_cls.shape))

        # 4. Extract l and C_l^TT for l=2 to lmax
        ls = np.arange(2, lmax_val + 1)
        
        if lmax_val >= all_cls.shape[0]:
            print("Error: lmax_val (" + str(lmax_val) + ") is out of bounds for the calculated spectra (max l=" + str(all_cls.shape[0] - 1) + ").")
            actual_lmax_available = all_cls.shape[0] - 1
            if actual_lmax_available < 2:
                 print("Error: Not enough multipoles available.")
                 return None, None
            ls = np.arange(2, min(lmax_val, actual_lmax_available) + 1)
            cl_TT = all_cls[2:min(lmax_val, actual_lmax_available) + 1, 0]
            print("Warning: Requested lmax " + str(lmax_val) + " was greater than available " + str(actual_lmax_available) + ". Truncating.")
        else:
            cl_TT = all_cls[2:lmax_val + 1, 0]

        print("Successfully extracted C_l^TT spectrum.")
        print("Shape of ls: " + str(ls.shape))
        print("Shape of cl_TT: " + str(cl_TT.shape))

        if ls.size > 0 and cl_TT.size > 0:
            print("\nSample of calculated C_l^TT values (in muK^2):")
            print("First 5 values:")
            for i in range(min(5, len(ls))):
                print("l=" + str(ls[i]) + ", TT=" + str(cl_TT[i]))
            
            if len(ls) > 5:
                print("...")
                print("Last 5 values:")
                for i in range(max(0, len(ls) - 5), len(ls)):
                    print("l=" + str(ls[i]) + ", TT=" + str(cl_TT[i]))
        else:
            print("No data to display for C_l^TT.")
            return None, None

        return ls, cl_TT

    except Exception as e:
        print("An error occurred during CAMB computation or data extraction:")
        import traceback
        print(traceback.format_exc())
        return None, None

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Created directory: " + data_dir)

    ls_mp, cl_TT_mp = calculate_cmb_power_spectrum()

    if ls_mp is not None and cl_TT_mp is not None:
        print("\nCMB power spectrum calculation successful.")

        # Basic validation check
        if len(ls_mp) > 0 and len(cl_TT_mp) > 0 and np.all(cl_TT_mp >= 0):
            print("Basic validation: Data seems reasonable (non-empty, C_l^TT >= 0).")
        else:
            print("Warning: Basic validation failed. Data might be problematic.")
            if len(ls_mp) == 0 or len(cl_TT_mp) == 0:
                 print("Reason: ls_mp or cl_TT_mp is empty.")
            if np.any(cl_TT_mp < 0):
                 print("Reason: Some C_l^TT values are negative.")


        # Format results into two columns: l and TT
        output_data = np.column_stack((ls_mp.astype(int), cl_TT_mp))
        
        # Save the data as result.csv
        csv_filename = os.path.join(data_dir, "result.csv")
        csv_header = "l,TT"
        try:
            np.savetxt(csv_filename, output_data, delimiter=',', header=csv_header, comments='', fmt=['%d', '%.8e'])
            print("CMB power spectrum (l, TT) saved to " + str(csv_filename))
        except Exception as e:
            print("Error saving data to CSV: " + str(csv_filename))
            print(e)

        # Create a simple visualization of the power spectrum
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(ls_mp, cl_TT_mp)
            plt.xlabel("Multipole moment l")
            plt.ylabel("C_l^TT (muK^2)") # C_l^TT in muK^2
            plt.title("CMB Temperature Power Spectrum")
            plt.yscale('log') # Use log scale for y-axis to see features
            plt.grid(True, which="both", ls="-")
            
            # Ensure plot elements are not overlapping
            plt.tight_layout()

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            plot_filename = os.path.join(data_dir, "cmb_power_spectrum_plot_1_" + timestamp + ".png")
            plt.savefig(plot_filename, dpi=300)
            plt.close() # Close the plot figure
            print("Plot of CMB power spectrum saved to " + str(plot_filename))
            print("Plot description: CMB Temperature Power Spectrum (C_l^TT in muK^2 vs. multipole moment l). Y-axis is log-scaled.")

        except Exception as e:
            print("Error generating or saving plot:")
            import traceback
            print(traceback.format_exc())

    else:
        print("\nCMB power spectrum calculation failed. Cannot proceed to save or plot.")
