# filename: codebase/cmb_power_spectrum.py
import camb
import numpy as np
import traceback  # For detailed error reporting
import matplotlib
matplotlib.rcParams['text.usetex'] = False  # Set globally to avoid LaTeX issues
import matplotlib.pyplot as plt
import pandas as pd
import os
import time


def calculate_cmb_power_spectrum():
    """
    Calculates the CMB raw temperature power spectrum (ClTT) for a flat Lambda CDM cosmology
    using specified parameters with CAMB.

    The function sets up cosmological parameters, runs CAMB, and extracts the unlensed
    scalar temperature power spectrum ClTT in units of muK^2 for multipole moments
    from l=2 to l=3000.

    Cosmological Parameters:
        Hubble constant (H0): 74 km/s/Mpc
        Baryon density (ombh2): 0.022 (dimensionless Omega_b * h^2)
        Cold dark matter density (omch2): 0.122 (dimensionless Omega_c * h^2)
        Neutrino mass sum (mnu): 0.06 eV
        Curvature (omk): 0 (dimensionless Omega_k)
        Optical depth to reionization (tau): 0.06 (dimensionless)
        Scalar amplitude (As): 2.0e-9 (dimensionless)
        Scalar spectral index (ns): 0.965 (dimensionless)
        Maximum multipole (lmax): 3000

    Returns:
        tuple: (ell, TT_spectrum)
            ell (numpy.ndarray): Array of multipole moments (l) from 2 to 3000.
            TT_spectrum (numpy.ndarray): Array of raw temperature power spectrum (ClTT) values in muK^2.
            Returns (None, None) if an error occurs.
    """
    # Define cosmological parameters
    H0 = 74.0  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Physical baryon density (Omega_b * h^2)
    omch2 = 0.122  # Physical cold dark matter density (Omega_c * h^2)
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature density (Omega_k)
    tau = 0.06  # Optical depth to reionization
    As = 2.0e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    lmax = 3000  # Maximum multipole moment for calculation

    # CAMB settings
    pars = camb.CAMBparams()

    try:
        # Set cosmological parameters
        # num_massive_neutrinos=3 assumes the sum mnu is distributed among 3 massive neutrino species.
        # Neff=3.046 is the standard effective number of relativistic species.
        pars.set_cosmology(
            H0=H0, 
            ombh2=ombh2, 
            omch2=omch2, 
            mnu=mnu, 
            omk=omk, 
            tau=tau, 
            num_massive_neutrinos=3, 
            Neff=3.046
        )

        # Set initial power spectrum parameters
        pars.InitPower.set_params(As=As, ns=ns)

        # Set calculation options for lmax
        # lens_potential_accuracy=0 for unlensed scalar spectra (raw spectra)
        # max_eta_k_scalar is increased for higher l accuracy, as per CAMB examples for lmax > 2000
        pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=0, max_eta_k_scalar=float(lmax*2.5)) 
        
        # Specify that we want scalar spectra and do not need tensor spectra for this task.
        pars.WantScalars = True
        pars.WantTensors = False 
        
        # Run CAMB calculation
        print("Running CAMB to calculate power spectra...")
        results = camb.get_results(pars)
        print("CAMB calculation finished.")

        # Get power spectra
        # CMB_unit='muK' ensures output Cls are in muK^2.
        # raw_cl=True ensures C_l rather than l(l+1)C_l/2pi.
        # lmax here specifies the maximum l to retrieve from the results.
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
        
        # Extract unlensed scalar Cls
        cls_unlensed_scalar = powers['unlensed_scalar']
        
        if cls_unlensed_scalar.shape[0] < lmax + 1:
            raise ValueError(
                "CAMB returned fewer multipoles than requested (lmax=" + str(lmax) + "). " +
                "Array shape: " + str(cls_unlensed_scalar.shape)
            )

        # Extract relevant part of the array: l from 2 to lmax
        ell = cls_unlensed_scalar[2:lmax+1, 0].astype(int)  # Multipole moment l
        TT_spectrum = cls_unlensed_scalar[2:lmax+1, 1]    # TT power spectrum (ClTT) in muK^2

        print("Successfully calculated CMB TT power spectrum.")
        print("Shape of l array (multipoles): " + str(ell.shape)) 
        print("Shape of TT spectrum array: " + str(TT_spectrum.shape))
        
        print("\nSample of calculated CMB TT power spectrum values (in muK^2):")
        print("First 5 values:")
        for i in range(min(5, len(ell))):
            print("l = " + str(ell[i]) + ", TT = " + str(TT_spectrum[i]))

        print("\nLast 5 values:")
        start_index_for_last_values = max(0, len(ell) - 5)
        for i in range(start_index_for_last_values, len(ell)):
            print("l = " + str(ell[i]) + ", TT = " + str(TT_spectrum[i]))
            
        return ell, TT_spectrum

    except Exception:
        print("An error occurred during CAMB calculation or data processing:")
        print(traceback.format_exc()) 
        return None, None


def plot_and_save_power_spectrum(ell, tt_spectrum, output_dir="data"):
    """
    Plots the CMB power spectrum and saves it as a PNG file.
    Saves the power spectrum data (l, TT) to a CSV file.

    Args:
        ell (numpy.ndarray): Array of multipole moments (l).
        tt_spectrum (numpy.ndarray): Array of raw temperature power spectrum (ClTT) values in muK^2.
        output_dir (str): Directory to save the plot and CSV file. Defaults to "data".
    """
    if ell is None or tt_spectrum is None or len(ell) == 0:
        print("Error: Input data for plotting/saving is invalid or empty.")
        return

    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        print("Ensured output directory exists: " + output_dir)
    except OSError as e:
        print("Error creating directory " + output_dir + ": " + str(e))
        return  # Cannot proceed without output directory

    # --- Plotting ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plot_filename = os.path.join(output_dir, "cmb_tt_power_spectrum_plot_1_" + timestamp + ".png")

    try:
        plt.figure(figsize=(10, 6))
        plt.plot(ell, tt_spectrum) 
        plt.xlabel("Multipole moment l (dimensionless)")
        plt.ylabel("Raw $C_l^{TT}$ ($\\mu K^2$)")
        plt.title("CMB Temperature Power Spectrum (Unlensed Scalar)")
        plt.xscale('log') 
        plt.yscale('log') 
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300)
        print("Plot saved to: " + plot_filename)
        print("Plot description: CMB Temperature Power Spectrum (unlensed scalar $C_l^{TT}$) vs. multipole moment l. Both axes are on a logarithmic scale.")
        plt.close()  # Close the figure to free memory
    except Exception as e:
        print("Error during plotting: " + str(e))
        print(traceback.format_exc())

    # --- Saving to CSV ---
    csv_filename = os.path.join(output_dir, "result.csv")
    try:
        df = pd.DataFrame({'l': ell.astype(int), 'TT': tt_spectrum})  # Ensure l is int for CSV
        df.to_csv(csv_filename, index=False, header=['l', 'TT']) 
        print("Power spectrum data saved to: " + csv_filename)

        # Verification (simple check)
        df_check = pd.read_csv(csv_filename)
        if 'l' in df_check.columns and 'TT' in df_check.columns:
            print("CSV file " + csv_filename + " verified: Columns 'l' and 'TT' are present.")
            if len(df_check) == len(ell):
                print("CSV file " + csv_filename + " verified: Row count matches expected (" + str(len(ell)) + " rows).")
            else:
                print("Warning: CSV file " + csv_filename + " row count mismatch. Expected: " + str(len(ell)) + ", Found: " + str(len(df_check)))
        else:
            print("Warning: CSV file " + csv_filename + " column verification failed. Check header. Found columns: " + str(list(df_check.columns)))
    except Exception as e:
        print("Error during CSV saving or verification: " + str(e))
        print(traceback.format_exc())


if __name__ == '__main__':
    print("Starting CMB power spectrum calculation and processing...")
    
    ell_values, tt_values = calculate_cmb_power_spectrum()

    if ell_values is not None and tt_values is not None:
        print("\nCMB power spectrum calculation successful.")
        if len(ell_values) > 0:
            print("Multipole moment l ranges from " + str(ell_values[0]) + " to " + str(ell_values[-1]) + ".")
        else:
            print("No multipole moments were generated (ell_values array is empty).")
        print("TT spectrum values (in muK^2) are available for further processing.")
        
        # Process and save results
        plot_and_save_power_spectrum(ell_values, tt_values, output_dir="data")
        
    else:
        print("\nCMB power spectrum calculation failed. Please check error messages above.")
    
    print("\nScript finished.")
