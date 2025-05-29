# filename: codebase/calculate_matter_power_spectrum.py
import os
import numpy as np
import pandas as pd
import camb
from camb.model import CAMBError


def calculate_matter_power_spectrum():
    r"""
    Calculates the relative difference in the linear matter power spectrum P(k)
    at redshift z=0 between normal and inverted neutrino hierarchy models.

    The calculation uses a flat Lambda CDM cosmology with specified parameters
    via CAMB. The results (k and relative difference) are saved to a CSV file.

    Cosmological Parameters:
    - Hubble constant (H0): 67.5 km/s/Mpc
    - Baryon density (ombh2): 0.022
    - Cold dark matter density (omch2): 0.122
    - Neutrino mass sum (sum_mnu): 0.11 eV
    - Scalar amplitude (As): 2e-9
    - Scalar spectral index (ns): 0.965

    k-range for P(k):
    - 200 evenly spaced values for k (in h/Mpc)
    - Range: 1e-4 h/Mpc to 2.0 h/Mpc

    Output:
    - CSV file 'data/result.csv' with columns:
        - 'k': Wavenumber (h/Mpc)
        - 'rel_diff': Relative difference (P_inverted / P_normal - 1)
    """

    # Cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Physical baryon density
    omch2 = 0.122  # Physical cold dark matter density
    sum_mnu = 0.11  # Sum of neutrino masses in eV
    As = 2.0e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    omk = 0.0  # Curvature parameter (0 for flat)
    
    # k-space parameters
    k_min_h_mpc = 1e-4  # Minimum k in h/Mpc
    k_max_h_mpc = 2.0   # Maximum k in h/Mpc
    num_k_points = 200
    # k values for P(k) calculation (linear spacing)
    # These are k in h/Mpc units
    k_vals_h_mpc = np.linspace(k_min_h_mpc, k_max_h_mpc, num_k_points)
    
    # CAMB requires kmax for internal calculations to be slightly larger than max k requested for output
    camb_kmax_h_mpc = k_max_h_mpc * 1.2 

    print("Setting up CAMB parameters...")

    # Base CAMB parameters object (will be copied and modified)
    pars_base = camb.CAMBparams()
    # General settings that apply to both hierarchies initially
    pars_base.InitPower.set_params(As=As, ns=ns)
    pars_base.set_matter_power(redshifts=[0.0], kmax=camb_kmax_h_mpc, k_hunit=True)
    pars_base.WantCls = False 
    pars_base.NonLinear = camb.model.NonLinear_none

    # --- Normal Hierarchy ---
    print("Calculating P(k) for Normal Neutrino Hierarchy...")
    pars_nh = pars_base.copy()
    # Set cosmology specific to Normal Hierarchy
    pars_nh.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omk, 
                          mnu=sum_mnu, neutrino_hierarchy='normal')
    # Re-apply InitPower and set_matter_power to ensure they are correctly set on the copied pars_nh
    # after set_cosmology, which might alter some derived parameter values or states.
    pars_nh.InitPower.set_params(As=As, ns=ns) 
    pars_nh.set_matter_power(redshifts=[0.0], kmax=camb_kmax_h_mpc, k_hunit=True)

    try:
        results_nh = camb.get_results(pars_nh)
        power_interpolator_nh = results_nh.get_matter_power_interpolator()
        pk_normal = power_interpolator_nh.P(0, k_vals_h_mpc)  # P(k) at z=0 in (Mpc/h)^3
    except CAMBError as e:
        print("CAMB calculation failed for Normal Hierarchy.")
        print("Error: " + str(e))
        return

    # --- Inverted Hierarchy ---
    print("Calculating P(k) for Inverted Neutrino Hierarchy...")
    pars_ih = pars_base.copy()
    # Set cosmology specific to Inverted Hierarchy
    pars_ih.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omk, 
                          mnu=sum_mnu, neutrino_hierarchy='inverted')
    # Re-apply InitPower and set_matter_power for pars_ih
    pars_ih.InitPower.set_params(As=As, ns=ns)
    pars_ih.set_matter_power(redshifts=[0.0], kmax=camb_kmax_h_mpc, k_hunit=True)

    try:
        results_ih = camb.get_results(pars_ih)
        power_interpolator_ih = results_ih.get_matter_power_interpolator()
        pk_inverted = power_interpolator_ih.P(0, k_vals_h_mpc)  # P(k) at z=0 in (Mpc/h)^3
    except CAMBError as e:
        print("CAMB calculation failed for Inverted Hierarchy.")
        print("Error: " + str(e))
        return

    print("Calculating relative difference...")
    # Relative difference: (P_inverted / P_normal) - 1
    if np.any(pk_normal == 0):
        print("Warning: Zero P(k) values found in normal hierarchy spectrum. Relative difference will contain inf/NaN.")
    elif np.any(pk_normal < 0):
         print("Warning: Negative P(k) values found in normal hierarchy spectrum. Relative difference might be problematic.")
    
    # Suppress runtime warnings for division by zero if pk_normal contains zeros
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_diff = (pk_inverted / pk_normal) - 1.0
        # Replace NaNs or Infs that might result from division by zero if necessary
        # For example, if pk_normal is 0 and pk_inverted is 0, rel_diff could be NaN.
        # If pk_normal is 0 and pk_inverted is non-zero, rel_diff could be Inf.
        # The problem implies avoiding the warning, not necessarily complex handling of 0/0 cases.
        # If P(k) is truly zero, the relative difference is ill-defined or infinite.
        # CAMB should produce positive P(k) for k > 0.

    # Create DataFrame
    df_results = pd.DataFrame({
        'k': k_vals_h_mpc,      # k in h/Mpc
        'rel_diff': rel_diff    # Relative difference
    })

    # Save to CSV
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_path = os.path.join(output_dir, "result.csv")
    df_results.to_csv(file_path, index=False)

    print("\n" + "Successfully calculated relative difference in matter power spectra.")
    print("Results saved to: " + file_path)

    # Print summary of results
    print("\n" + "First 5 rows of the results:")
    print(df_results.head().to_string())

    print("\n" + "Summary statistics for the relative difference:")
    print(df_results['rel_diff'].describe().to_string())


if __name__ == '__main__':
    calculate_matter_power_spectrum()