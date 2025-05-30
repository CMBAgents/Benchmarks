# filename: codebase/cmb_power_spectrum.py
import numpy as np
import camb
import os
import matplotlib
matplotlib.rcParams['text.usetex'] = False # Disable LaTeX rendering
import matplotlib.pyplot as plt
import time

def calculate_cmb_power_spectrum():
    """
    Calculates the CMB raw temperature power spectrum using CAMB for a specific
    set of cosmological parameters.

    The function sets up the cosmological model, runs CAMB, and extracts
    the unlensed scalar temperature power spectrum (Cl^TT) in units of muK^2
    for multipole moments l = 2 to l = 3000.

    Cosmological Parameters:
        Hubble constant (H0): 74 km/s/Mpc
        Baryon density (ombh2): 0.022
        Cold dark matter density (omch2): 0.122
        Neutrino mass sum (mnu): 0.06 eV
        Curvature (omk): 0 (flat universe)
        Optical depth to reionization (tau): 0.06
        Scalar amplitude (As): 2e-9
        Scalar spectral index (ns): 0.965

    Returns:
        tuple: A tuple containing:
            - ls (np.ndarray): Array of multipole moments (l) from 2 to 3000.
            - cl_TT_selected (np.ndarray): Array of raw temperature power
                                           spectrum (Cl^TT) in muK^2.
                                           Returns (None, None) if CAMB fails.
    """
    # Cosmological parameters
    H0 = 74.0  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Physical baryon density (Omega_b * h^2)
    omch2 = 0.122  # Physical cold dark matter density (Omega_c * h^2)
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature density (Omega_k)
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude (at k=0.05 Mpc^-1)
    ns = 0.965  # Scalar spectral index
    lmax_calc = 3000 # Maximum multipole moment for CAMB calculation

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    try:
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
        pars.InitPower.set_params(As=As, ns=ns)
        # We want unlensed raw TT, so set lens_potential_accuracy=0
        pars.set_for_lmax(lmax=lmax_calc, lens_potential_accuracy=0) 
        
        # Calculate results
        print("Running CAMB to calculate power spectra...")
        results = camb.get_results(pars)
        print("CAMB calculation successful.")

        # Get unlensed scalar CMB power spectra in muK^2
        # The output is (lmax+1, 4) for TT, EE, BB, TE
        # Columns are TT, EE, BB, TE. Index is L.
        # CMB_unit='muK' ensures C_l is in muK^2
        unlensed_cls = results.get_unlensed_scalar_cls(lmax=lmax_calc, CMB_unit='muK')
        
        # Extract Cl^TT (first column)
        # unlensed_cls[l,0] is C_l^TT
        cl_TT_all = unlensed_cls[:, 0]  # Units: muK^2

        # We need l from 2 to 3000
        # CAMB output array is indexed from l=0.
        # So, for l=2, index is 2. For l=3000, index is 3000.
        
        l_min_output = 2
        l_max_output = 3000 # Inclusive
        
        if l_max_output > lmax_calc:
            print("Error: Requested l_max_output (" + str(l_max_output) + ") is greater than calculated lmax_calc (" + str(lmax_calc) + ")")
            return None, None

        # Create array of multipole moments 'l' from l_min_output to l_max_output
        ls = np.arange(l_min_output, l_max_output + 1) 
        # Select corresponding C_l^TT values.
        # cl_TT_all is indexed from l=0, so cl_TT_all[l] corresponds to C_l.
        cl_TT_selected = cl_TT_all[l_min_output : l_max_output + 1] # Units: muK^2

        print("Successfully computed C_l^TT power spectrum.")
        print("Shape of ls array: " + str(ls.shape))
        print("Shape of cl_TT_selected array: " + str(cl_TT_selected.shape))
        
        # Print a few values for verification
        print("\nFirst 5 values of l and C_l^TT (muK^2):")
        for i in range(min(5, len(ls))):
            print("l=" + str(ls[i]) + ", TT=" + str(cl_TT_selected[i]))

        print("\nLast 5 values of l and C_l^TT (muK^2):")
        for i in range(max(0, len(ls) - 5), len(ls)):
            print("l=" + str(ls[i]) + ", TT=" + str(cl_TT_selected[i]))
            
        return ls, cl_TT_selected

    except Exception as e:
        print("Error during CAMB calculation or data extraction:")
        print(str(e))
        return None, None

if __name__ == '__main__':
    database_path = 'data'
    if not os.path.exists(database_path):
        os.makedirs(database_path)
        print("Created directory: " + database_path)

    ls_values, cl_TT_values = calculate_cmb_power_spectrum()

    if ls_values is not None and cl_TT_values is not None:
        print("\nCMB power spectrum calculation completed.")
        
        # Save results to CSV
        output_csv_filename = os.path.join(database_path, 'result.csv')
        try:
            # Prepare data for CSV: two columns 'l' and 'TT'
            output_data = np.column_stack((ls_values, cl_TT_values))
            # Save to CSV with header
            np.savetxt(output_csv_filename, output_data, header='l,TT', delimiter=',', comments='')
            print("Results saved to " + output_csv_filename)
            
            # Verify CSV content (first few lines)
            with open(output_csv_filename, 'r') as f:
                print("\nFirst 5 lines of " + output_csv_filename + ":")
                for i in range(5):
                    line = f.readline().strip()
                    if not line:
                        break
                    print(line)

        except Exception as e:
            print("Error saving results to CSV: " + str(e))

        # Create and save a plot for verification
        try:
            # Calculate D_l^TT = l(l+1)C_l^TT / (2*pi)
            # This quantity is often plotted for CMB spectra.
            dl_TT_values = ls_values * (ls_values + 1) * cl_TT_values / (2 * np.pi) # Units: muK^2

            plt.figure(figsize=(10, 6))
            plt.plot(ls_values, dl_TT_values)
            plt.xlabel("Multipole moment l")
            plt.ylabel("D_l^TT = l(l+1)C_l^TT/(2pi) [muK^2]")
            plt.title("CMB Temperature Power Spectrum (Unlensed)")
            plt.grid(True)
            plt.tight_layout() # Adjust layout to prevent overlapping labels

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            plot_filename = os.path.join(database_path, 'cmb_power_spectrum_plot_1_' + timestamp + '.png')
            plt.savefig(plot_filename, dpi=300)
            print("\nPlot of the CMB power spectrum saved to: " + plot_filename)
            print("Description: The plot shows D_l^TT = l(l+1)C_l^TT/(2pi) vs. multipole moment l. " +
                  "D_l^TT is in units of muK^2. The spectrum displays characteristic acoustic peaks.")
            plt.close() # Close the plot to free memory

        except Exception as e:
            print("Error generating or saving plot: " + str(e))
            
    else:
        print("\nCMB power spectrum calculation failed. Cannot save results or plot.")
