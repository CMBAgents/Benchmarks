# filename: codebase/calculate_linear_matter_power_spectrum.py
import os
import numpy as np
import pandas as pd
import camb  # This will raise ImportError if camb is not installed.


def calculate_linear_matter_power_spectrum():
    r"""
    Calculates the linear matter power spectrum P(k) at redshift z=0
    for a flat Lambda CDM cosmology using specified parameters with CAMB.

    The cosmological parameters are:
    - Hubble constant (H0): 67.5 km/s/Mpc
    - Baryon density (Omega_b * h^2): 0.022
    - Cold dark matter density (Omega_c * h^2): 0.122
    - Sum of neutrino masses (Sigma m_nu): 0.06 eV
    - Curvature density (Omega_k): 0 (flat universe)
    - Optical depth to reionization (tau): 0.06
    - Scalar amplitude (A_s): 2.0 * 10^-9
    - Scalar spectral index (n_s): 0.965
    - Maximum k for CAMB's internal calculation (k_max): 2.0 h/Mpc

    The function computes P(k) for 200 evenly (linearly) spaced k/h values
    in the range [1e-4, 1.0] h/Mpc.

    The results are saved in a CSV file named 'result.csv' in the 'data/'
    directory, with two columns:
    - 'kh': Wavenumber k/h (units: h/Mpc)
    - 'P_k': Linear matter power spectrum P(k) (units: (Mpc/h)^3)
    """

    # Cosmological parameters
    H0_val = 67.5  # km/s/Mpc
    ombh2_val = 0.022
    omch2_val = 0.122
    mnu_val = 0.06  # eV
    omk_val = 0.0
    tau_val = 0.06
    As_val = 2e-9
    ns_val = 0.965
    kmax_camb = 2.0  # h/Mpc, for CAMB's internal calculation accuracy

    # Output k/h range
    kh_min_out = 1e-4  # h/Mpc
    kh_max_out = 1.0   # h/Mpc
    n_points_out = 200

    # --- CAMB Setup ---
    pars = camb.CAMBparams()

    pars.set_cosmology(
        H0=H0_val,
        ombh2=ombh2_val,
        omch2=omch2_val,
        mnu=mnu_val,
        omk=omk_val,
        tau=tau_val
    )

    pars.InitPower.set_params(As=As_val, ns=ns_val)

    # We need the linear matter power spectrum at z=0.
    # kmax here is the maximum k (in h/Mpc) up to which CAMB computes the spectrum.
    # This needs to be at least kh_max_out for accurate interpolation.
    pars.set_matter_power(redshifts=[0.], kmax=kmax_camb)

    # Ensure linear power spectrum is computed.
    # This is the default behavior if NonLinear is not otherwise set.
    pars.NonLinear = camb.model.NonLinear_none

    # --- Get Results from CAMB ---
    results = camb.get_results(pars)

    # --- Generate k/h values and interpolate P(k) ---
    # Create an array of 200 linearly spaced k/h values.
    # kh_values are in h/Mpc.
    kh_values = np.linspace(kh_min_out, kh_max_out, n_points_out)  # units: h/Mpc

    # Get the matter power spectrum interpolator object.
    # For linear P(k), ensure nonlinear=False (default if pars.NonLinear is NonLinear_none).
    # Hubble units (k in h/Mpc, P(k) in (Mpc/h)^3) are default.
    PK_interpolator = results.get_matter_power_interpolator()

    # Interpolate P(k) at z=0 for each k/h value.
    # P_k_values will be in (Mpc/h)^3.
    P_k_values = PK_interpolator.P(0, kh_values)  # z=0, kh_values are k/h in h/Mpc

    # --- Prepare and Save Data ---
    output_df = pd.DataFrame({
        'kh': kh_values,  # units: h/Mpc
        'P_k': P_k_values  # units: (Mpc/h)^3
    })

    # Create the 'data' directory if it doesn't exist
    data_directory = 'data'
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        print("Created directory: " + data_directory)

    # Define file path and save to CSV
    csv_file_path = os.path.join(data_directory, 'result.csv')
    output_df.to_csv(csv_file_path, index=False)

    # --- Print Summary to Console ---
    print("Successfully calculated the linear matter power spectrum.")
    print("Results saved to: " + csv_file_path)
    print("\nSummary of the results:")
    print("Number of k/h points: " + str(len(output_df)))
    print("k/h range (h/Mpc): [" + str(kh_values.min()) + ", " + str(kh_values.max()) + "]")
    
    # Set pandas options to display more content from the DataFrame
    # This ensures the head and tail are printed in full, not truncated.
    # Using to_string() method for DataFrame printing is more robust.
    print("\nFirst 5 rows of the data:")
    print(output_df.head(5).to_string())
    print("\nLast 5 rows of the data:")
    print(output_df.tail(5).to_string())


if __name__ == '__main__':
    try:
        calculate_linear_matter_power_spectrum()
    except Exception as e:
        # Catch any exception during the execution of the main function
        print("An error occurred during the calculation:")
        # Print the full traceback for detailed error information
        import traceback
        print(traceback.format_exc())