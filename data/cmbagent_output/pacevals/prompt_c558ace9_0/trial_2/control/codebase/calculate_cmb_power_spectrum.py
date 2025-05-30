# filename: codebase/calculate_cmb_power_spectrum.py
import numpy as np
import camb
import os

def calculate_cmb_power_spectrum():
    """
    Calculates the CMB raw temperature power spectrum using CAMB for a specific
    set of cosmological parameters.

    The function sets up the cosmological model, runs CAMB, and extracts
    the unlensed scalar temperature power spectrum (Cl^TT) in units of muK^2
    for multipole moments l = 2 to l = 3000.

    Returns:
        tuple: A tuple containing:
            - ls (np.ndarray): Array of multipole moments (l) from 2 to 3000.
            - cl_TT_selected (np.ndarray): Array of raw temperature power
                                           spectrum (Cl^TT) in muK^2.
                                           Returns (None, None) if CAMB fails.
    """
    # Cosmological parameters
    H0 = 74.0  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Physical baryon density
    omch2 = 0.122  # Physical cold dark matter density
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature density
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    lmax_calc = 3000 # Maximum multipole moment for calculation

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    try:
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
        pars.InitPower.set_params(As=As, ns=ns)
        pars.set_for_lmax(lmax=lmax_calc, lens_potential_accuracy=0) # lens_potential_accuracy=0 for unlensed
        
        # Calculate results
        print("Running CAMB to calculate power spectra...")
        results = camb.get_results(pars)
        print("CAMB calculation successful.")

        # Get unlensed scalar CMB power spectra in muK^2
        # The output is (lmax+1, 4) for TT, EE, BB, TE
        # Columns are TT, EE, BB, TE. Index is L.
        unlensed_cls = results.get_unlensed_scalar_cls(lmax=lmax_calc, CMB_unit='muK')
        
        # Extract Cl^TT (first column)
        # unlensed_cls[l,0] is C_l^TT
        cl_TT_all = unlensed_cls[:, 0]  # Units: muK^2

        # We need l from 2 to 3000
        # CAMB output array is indexed from l=0.
        # So, for l=2, index is 2. For l=3000, index is 3000.
        # The length of cl_TT_all is lmax_calc + 1.
        
        l_min = 2
        l_max = 3000 # Inclusive
        
        if l_max > lmax_calc:
            print("Error: Requested l_max (" + str(l_max) + ") is greater than calculated lmax_calc (" + str(lmax_calc) + ")")
            return None, None

        ls = np.arange(l_min, l_max + 1) # Multipole moments from 2 to 3000
        cl_TT_selected = cl_TT_all[l_min : l_max + 1] # Corresponding C_l^TT values

        print("Successfully computed C_l^TT power spectrum.")
        print("Shape of ls array: " + str(ls.shape))
        print("Shape of cl_TT_selected array: " + str(cl_TT_selected.shape))
        
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
    # Create data directory if it doesn't exist
    database_path = 'data'
    if not os.path.exists(database_path):
        os.makedirs(database_path)
        print("Created directory: " + database_path)

    ls_values, cl_TT_values = calculate_cmb_power_spectrum()

    if ls_values is not None and cl_TT_values is not None:
        print("\nCMB power spectrum calculation completed.")
        # Further steps like saving to CSV or plotting would go here
        # For now, we just confirm the calculation ran.
        
        # Example: Save to CSV (this part will be formally done in the next step)
        output_filename = os.path.join(database_path, 'result.csv')
        try:
            output_data = np.column_stack((ls_values, cl_TT_values))
            np.savetxt(output_filename, output_data, header='l,TT', delimiter=',', comments='')
            print("Results saved to " + output_filename)
        except Exception as e:
            print("Error saving results to CSV: " + str(e))
            
    else:
        print("\nCMB power spectrum calculation failed.")
