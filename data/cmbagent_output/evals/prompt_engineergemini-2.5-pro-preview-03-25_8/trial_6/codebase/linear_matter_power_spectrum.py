# filename: codebase/linear_matter_power_spectrum.py
import os
import numpy as np
import pandas as pd
try:
    import camb
except ImportError:
    print("CAMB is not installed. Please install it using 'pip install camb'")
    # In a real workflow, another agent would handle this.
    # For standalone execution, one might exit here or raise the error.
    raise


def calculate_matter_power_spectrum():
    r"""
    Calculates the linear matter power spectrum using CAMB for a specified cosmology.

    The function sets up cosmological parameters, computes the power spectrum at z=0
    for a range of k values, and saves the results to a CSV file.

    Cosmological Parameters:
    - Hubble constant (H0): 67.5 km/s/Mpc
    - Baryon density (ombh2): 0.022
    - Cold dark matter density (omch2): 0.122
    - Neutrino mass sum (mnu): 0.06 eV
    - Curvature (omk): 0
    - Optical depth to reionization (tau): 0.06
    - Scalar amplitude (As): 2e-9
    - Scalar spectral index (ns): 0.965
    - k_max for CAMB internal calculation: 2.0 Mpc^-1

    Output k-range:
    - kh: 200 evenly spaced values from 1e-4 to 1.0 (k/h, units Mpc^-1)
    - P_k: Linear matter power spectrum in (Mpc/h)^3
    """

    # Cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Physical baryon density
    omch2 = 0.122  # Physical cold dark matter density
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature density
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    kmax_camb_internal = 2.0  # k_max for CAMB's internal calculations, in Mpc^-1

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    
    # We want the linear power spectrum at z=0
    # kmax here is the maximum k (in Mpc^-1) for CAMB's internal calculation of transfer functions
    pars.set_matter_power(redshifts=[0.], kmax=kmax_camb_internal)
    
    # Tell CAMB to calculate the linear power spectrum
    # (NonLinear_none means linear computation, which is default for get_matter_power_interpolator if nonlinear=False)
    # pars.NonLinear = camb.model.NonLinear_none # This is default if not asking for non-linear.

    # Get results from CAMB
    results = camb.get_results(pars)

    # Define the k/h range for output
    # 200 evenly spaced k values in the range 10^-4 < kh < 1 (Mpc^-1)
    # kh_values are k/h, where k is in Mpc^-1 and h is H0/100. Unit of kh_values is Mpc^-1.
    kh_min = 1e-4  # min k/h (Mpc^-1)
    kh_max = 1.0  # max k/h (Mpc^-1)
    n_points = 200
    kh_values = np.linspace(kh_min, kh_max, n_points)  # Array of k/h values (Mpc^-1)

    # Get the matter power spectrum interpolator for P_L(k)
    # hubble_units=True means P(k) is in (Mpc/h)^3
    # k_hunit=True means input k values to PK.P are k/h
    # nonlinear=False ensures we get the linear power spectrum
    PK_interpolator = results.get_matter_power_interpolator(nonlinear=False, hubble_units=True, k_hunit=True)

    # Calculate P(k) at z=0 for the specified k/h values
    # P_k_values will be in (Mpc/h)^3
    P_k_values = PK_interpolator.P(0, kh_values)

    # Create a pandas DataFrame
    df_results = pd.DataFrame({
        'kh': kh_values,  # Wavenumber k/h (Mpc^-1)
        'P_k': P_k_values  # Linear matter power spectrum P(k) ((Mpc/h)^3)
    })

    # Create data directory if it doesn't exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Created directory: " + data_dir)

    # Save results to CSV
    file_path = os.path.join(data_dir, 'result.csv')
    df_results.to_csv(file_path, index=False)

    print("Linear matter power spectrum calculation complete.")
    print("Results saved to: " + file_path)
    print("First few rows of the data:")
    # Configure pandas to display float with more precision if needed for print
    pd.set_option('display.float_format', '{:.6e}'.format)
    print(df_results.head())


if __name__ == '__main__':
    calculate_matter_power_spectrum()