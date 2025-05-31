# filename: codebase/calculate_cmb_power_spectrum.py
import os
import numpy as np
try:
    import camb
except ImportError:
    print("CAMB package not found. Please install it, e.g., using 'pip install camb'.")
    # In a real scenario, another agent would handle installation.
    # For this exercise, we'll exit if CAMB is not found.
    exit()

def calculate_cmb_power_spectrum():
    """
    Calculates the CMB raw temperature power spectrum for a flat Lambda CDM cosmology
    using CAMB and saves the results to a CSV file.

    The function sets up cosmological parameters, runs CAMB, extracts the
    TT power spectrum, converts it to microKelvin^2 units, and saves
    it for multipole moments l=2 to l=3000.
    """
    # Define cosmological parameters
    h0 = 74.0  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Physical baryon density
    omch2 = 0.122  # Physical cold dark matter density
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    lmax_calc = 3000 # Maximum multipole moment to calculate

    print("Setting up CAMB parameters...")
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk)
    pars.Reion.optical_depth = tau
    pars.InitPower.As = As
    pars.InitPower.ns = ns
    
    # We want unlensed raw spectra, so lens_potential_accuracy=0
    pars.set_for_lmax(lmax=lmax_calc, lens_potential_accuracy=0)
    
    # Suppress CAMB's own output where possible if it's too verbose
    # (Actual suppression depends on CAMB's internal logging, this is a general idea)
    # pars.WantTransfer = True # Example of a parameter, not directly for verbosity

    print("Cosmological parameters set:")
    print("H0: " + str(h0) + " km/s/Mpc")
    print("ombh2: " + str(ombh2))
    print("omch2: " + str(omch2))
    print("mnu: " + str(mnu) + " eV")
    print("omk: " + str(omk))
    print("tau: " + str(tau))
    print("As: " + str(As))
    print("ns: " + str(ns))
    print("lmax: " + str(lmax_calc))

    try:
        print("\nRunning CAMB calculations...")
        results = camb.get_results(pars)
        print("CAMB calculations finished.")

        # Get unlensed scalar Cls
        # These are C_l/TCMB^2, where TCMB is in Kelvin.
        # The output array has columns: TT, EE, BB, TE
        # It is indexed from l=0 to lmax_calc.
        cls_unlensed_scalar = results.get_unlensed_scalar_cls(lmax=lmax_calc)
        
        # Extract TT spectrum (C_l^TT / TCMB^2)
        cl_tt_dimensionless = cls_unlensed_scalar[:, 0] # Units: dimensionless (factor of TCMB^2 removed)

        # Convert to muK^2
        # TCMB is in Kelvin, pars.TCMB stores this value (default 2.7255 K)
        tcmb_kelvin = pars.TCMB # Units: K
        tcmb_muk = tcmb_kelvin * 1e6 # Units: muK
        
        # C_l^TT [muK^2] = (C_l^TT / TCMB_K^2) * (TCMB_muK^2)
        # C_l^TT [muK^2] = (C_l^TT / TCMB_K^2) * (TCMB_K * 1e6)^2
        # C_l^TT [muK^2] = (C_l^TT / TCMB_K^2) * TCMB_K^2 * (1e6)^2 -> This is wrong.
        # Correct: C_l [K^2] = (C_l / Tcmb^2) * Tcmb^2
        # C_l [muK^2] = C_l [K^2] * (1e6)^2
        # So, C_l [muK^2] = (C_l / Tcmb^2) * Tcmb^2 * (1e6)^2 is not what we want.
        # We have C_l/T_0^2. We want C_l in (muK)^2.
        # C_l^{TT} (output from CAMB) is D_l^{TT} / (T_CMB^0)^2 where D_l is the power spectrum in K^2.
        # So, D_l^{TT} = C_l^{TT} * (T_CMB^0)^2.  Units: K^2
        # To get to muK^2, multiply by (10^6)^2.
        # D_l^{TT} (muK^2) = C_l^{TT} * (T_CMB^0)^2 * (10^6)^2 = C_l^{TT} * (T_CMB^0 * 10^6)^2
        # This is equivalent to: cl_tt_muK2 = cl_tt_dimensionless * (tcmb_muk**2)
        
        cl_tt_muK2 = cl_tt_dimensionless * (tcmb_muk**2) # Units: muK^2

        # We need l from 2 to 3000.
        # cls_unlensed_scalar is indexed from l=0.
        # So, index 2 corresponds to l=2.
        l_start_index = 2
        l_end_index = lmax_calc 
        
        # Multipole moments from l=2 to l=lmax_calc
        l_values = np.arange(l_start_index, l_end_index + 1) # Integer values
        
        # Corresponding C_l^TT values
        cl_tt_muK2_selected = cl_tt_muK2[l_start_index : l_end_index + 1] # Units: muK^2

        # Validation
        expected_rows = lmax_calc - l_start_index + 1
        if cl_tt_muK2_selected.shape[0] != expected_rows:
            print("Error: Unexpected number of rows in selected C_l^TT data.")
            print("Expected: " + str(expected_rows) + ", Got: " + str(cl_tt_muK2_selected.shape[0]))
            return

        if not np.all(cl_tt_muK2_selected >= 0):
            print("Warning: Some C_l^TT values are negative, which is unphysical.")
            # This might indicate an issue with parameters or calculation if unexpected.
            # For raw TT, they should be positive.

        # Prepare data for saving
        output_data = np.column_stack((l_values, cl_tt_muK2_selected))

        # Create data directory if it doesn't exist
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print("Created directory: " + data_dir)

        output_filename = os.path.join(data_dir, "result.csv")
        header_str = "l,TT"  # TT is C_l^TT in muK^2

        np.savetxt(output_filename, output_data, header=header_str, delimiter=',' , comments='', fmt=['%d', '%.18e'])
        
        print("\nCMB temperature power spectrum (C_l^TT) saved to: " + output_filename)
        print("Shape of the saved data (rows, columns): " + str(output_data.shape))
        
        print("\nFirst 5 entries (l, TT [muK^2]):")
        for i in range(min(5, output_data.shape[0])):
            print(str(int(output_data[i,0])) + ", " + str(output_data[i,1]))

        if output_data.shape[0] > 5:
            print("\nLast 5 entries (l, TT [muK^2]):")
            for i in range(max(0, output_data.shape[0]-5), output_data.shape[0]):
                 print(str(int(output_data[i,0])) + ", " + str(output_data[i,1]))

    except Exception as e:
        print("An error occurred during CAMB calculation or data processing:")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    calculate_cmb_power_spectrum()
