# filename: codebase/cmb_power_spectrum.py
import camb
import numpy as np
import os

def calculate_cmb_power_spectrum():
    """
    Calculates the CMB raw temperature power spectrum for a flat Lambda CDM cosmology
    using specified parameters with CAMB and saves the results to a CSV file.
    """
    # Define cosmological parameters
    H0 = 74.0  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Physical baryon density
    omch2 = 0.122  # Physical cold dark matter density
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature parameter
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    lmax_calc = 3000 # Maximum multipole moment for calculation

    # 1. Initialize CAMB Parameters
    pars = camb.CAMBparams()

    # 2. Set Cosmological Parameters
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)

    # 3. Set Initial Power Spectrum Parameters
    pars.InitPower.set_params(As=As, ns=ns)

    # 4. Set Multipole Range for calculation
    # lens_potential_accuracy=0 is specified to match the example,
    # though for unlensed spectra it's less critical.
    # CAMB calculates up to lmax.
    pars.set_for_lmax(lmax=lmax_calc, lens_potential_accuracy=0)

    # 5. Run CAMB Calculations
    print("Running CAMB to calculate power spectra...")
    results = camb.get_results(pars)
    print("CAMB calculation complete.")

    # 6. Extract Raw Temperature Power Spectrum (C_l^TT)
    # We want C_l, so raw_cl=True.
    # We want unlensed scalar spectra.
    # Units: muK^2
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True, lmax=lmax_calc, spectra=['unlensed_scalar'])
    
    # powers['unlensed_scalar'] is an array (lmax+1, n_spectra)
    # For 'unlensed_scalar', n_spectra is 1 (TT) if only temperature is requested or computed by default for this.
    # More generally, it can be [TT, EE, BB, TE] if those are computed.
    # For raw_cl=True and spectra=['unlensed_scalar'], it should be (lmax+1,1) for TT if get_scalar_cls=True (default)
    # Or, if it computes all [TT, EE, BB, TE], then cl_TT is the first column.
    # Based on CAMB docs, for 'unlensed_scalar', the first column is TT.
    cl_TT_all = powers['unlensed_scalar'][:, 0]  # TT spectrum, from l=0 to l=lmax_calc

    # 7. Prepare Data for CSV (l from 2 to 3000)
    l_min_output = 2
    l_max_output = 3000
    
    # Ensure l_max_output does not exceed lmax_calc
    if l_max_output > lmax_calc:
        print("Warning: Requested l_max_output (" + str(l_max_output) + ") exceeds lmax_calc (" + str(lmax_calc) + "). Truncating to lmax_calc.")
        l_max_output = lmax_calc
        
    l_values = np.arange(l_min_output, l_max_output + 1)
    
    # Select the C_l^TT values for the desired l_values range
    # cl_TT_all is indexed from l=0. So, cl_TT_all[l] corresponds to multipole l.
    TT_powers = cl_TT_all[l_values]
    
    output_data = np.column_stack((l_values, TT_powers))

    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Created directory: " + data_dir)

    file_path = os.path.join(data_dir, "result.csv")
    np.savetxt(file_path, output_data, header='l,TT', delimiter=',', comments='')
    print("CMB temperature power spectrum saved to: " + file_path)

    # Print some results to console
    print("\nCMB Raw Temperature Power Spectrum (C_l^TT) in muK^2:")
    print("----------------------------------------------------")
    print("l          TT (muK^2)")
    print("----------------------------------------------------")
    for i in range(min(5, len(output_data))):
        print(str(int(output_data[i, 0])) + "          " + str(output_data[i, 1]))
    
    if len(output_data) > 10:
        print("...")
        for i in range(max(len(output_data) - 5, 5), len(output_data)):
             print(str(int(output_data[i, 0])) + "          " + str(output_data[i, 1]))
    elif len(output_data) > 5 :
        for i in range(5, len(output_data)):
             print(str(int(output_data[i, 0])) + "          " + str(output_data[i, 1]))
    print("----------------------------------------------------")
    print("Total " + str(len(output_data)) + " data points saved.")

if __name__ == '__main__':
    try:
        calculate_cmb_power_spectrum()
    except ImportError:
        print("CAMB Python package not found. Please install it first.")
        print("You can typically install it using: pip install camb")
    except Exception as e:
        print("An error occurred during the CAMB calculation:")
        import traceback
        traceback.print_exc()
