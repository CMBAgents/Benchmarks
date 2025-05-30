# filename: codebase/power_spectrum.py
import os
import camb
import numpy as np
import pandas as pd

def get_pk(hierarchy_str, H0_val, ombh2_val, omch2_val, mnu_sum_val, As_val, ns_val,
           k_values_h_mpc, kmax_config_mpc_inv, redshift_val=0.0):
    """
    Computes the linear matter power spectrum P(k/h) using CAMB.

    Parameters:
    ----------
    hierarchy_str : str
        Neutrino hierarchy, 'normal' or 'inverted'.
    H0_val : float
        Hubble constant (km/s/Mpc).
    ombh2_val : float
        Baryon density parameter.
    omch2_val : float
        Cold dark matter density parameter.
    mnu_sum_val : float
        Sum of neutrino masses (eV).
    As_val : float
        Scalar amplitude.
    ns_val : float
        Scalar spectral index.
    k_values_h_mpc : np.ndarray
        Array of k/h values (h/Mpc) at which to compute P(k).
    kmax_config_mpc_inv : float
        Maximum k (Mpc^-1) for CAMB's internal calculation range.
    redshift_val : float, optional
        Redshift at which to compute P(k). Default is 0.0.

    Returns:
    -------
    np.ndarray
        Linear matter power spectrum P(k/h) in (Mpc/h)^3.
        Returns None if CAMB computation fails.
    """
    try:
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0_val, ombh2=ombh2_val, omch2=omch2_val, mnu=mnu_sum_val, omk=0,
                          neutrino_hierarchy=hierarchy_str, num_massive_neutrinos=3, Neff=3.046)
        pars.InitPower.set_params(As=As_val, ns=ns_val)
        pars.set_matter_power(redshifts=[redshift_val], kmax=kmax_config_mpc_inv, 
                              accurate_massive_neutrino_transfers=True)
        
        results = camb.get_results(pars)
        
        # Get linear matter power spectrum interpolator
        # P(k) will be P(k/h) in (Mpc/h)^3, k will be k/h in h/Mpc
        pk_interp = results.get_matter_power_interpolator(nonlinear=False, hubble_units=True, k_hunit=True)
        
        pk_values = pk_interp.P(redshift_val, k_values_h_mpc)
        return pk_values
    except Exception as e:
        print("Error during CAMB computation for " + hierarchy_str + " hierarchy:")
        print(str(e))
        return None


def main():
    """
    Main function to calculate and save the relative difference in matter power spectra.
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant (km/s/Mpc)
    ombh2 = 0.022  # Baryon density
    omch2 = 0.122  # Cold dark matter density
    mnu_sum = 0.11  # Neutrino mass sum (eV)
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    
    h_param = H0 / 100.0 # Dimensionless Hubble parameter
    
    # k range definition
    # k_physical is k in Mpc^-1
    # k_values_for_csv are k/h in h/Mpc
    k_physical_min_mpc_inv = 1e-4  # Mpc^-1
    k_physical_max_mpc_inv = 2.0   # Mpc^-1
    num_k_points = 200
    
    # k_values for interpolation are k/h (h/Mpc)
    k_over_h_min = k_physical_min_mpc_inv / h_param
    k_over_h_max = k_physical_max_mpc_inv / h_param
    
    k_values_for_interp = np.linspace(k_over_h_min, k_over_h_max, num_k_points) # units: h/Mpc
    
    # kmax for CAMB configuration (in Mpc^-1)
    # Should be somewhat larger than the maximum physical k we query (k_physical_max_mpc_inv)
    kmax_for_camb_config = k_physical_max_mpc_inv + h_param

    print("Calculating P(k) for normal hierarchy...")
    pk_normal = get_pk('normal', H0, ombh2, omch2, mnu_sum, As, ns, 
                       k_values_for_interp, kmax_for_camb_config)
    
    if pk_normal is None:
        print("Failed to compute P(k) for normal hierarchy. Exiting.")
        return

    print("Calculating P(k) for inverted hierarchy...")
    pk_inverted = get_pk('inverted', H0, ombh2, omch2, mnu_sum, As, ns,
                         k_values_for_interp, kmax_for_camb_config)

    if pk_inverted is None:
        print("Failed to compute P(k) for inverted hierarchy. Exiting.")
        return

    # Calculate relative difference: (P_inverted / P_normal - 1)
    # Handle potential division by zero or pk_normal being very small
    # np.divide will output np.nan where pk_normal is zero, due to out=np.full_like(...)
    # This avoids RuntimeWarning: invalid value encountered in divide / divide by zero
    rel_diff = np.full_like(pk_normal, np.nan)  # Initialize with NaNs
    
    # Perform division only where pk_normal is not zero
    valid_indices = pk_normal != 0
    rel_diff[valid_indices] = pk_inverted[valid_indices] / pk_normal[valid_indices] - 1.0
    
    # Create DataFrame
    df_results = pd.DataFrame({
        'k': k_values_for_interp,  # k in h/Mpc
        'rel_diff': rel_diff
    })

    # Save results to CSV
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_path = os.path.join(output_dir, "result.csv")
    df_results.to_csv(file_path, index=False)
    
    print("\nResults saved to: " + file_path)

    # Print summary of relative differences
    print("\nSummary of relative differences (P_inverted / P_normal - 1):")
    # Configure pandas to display more precision
    pd.set_option('display.float_format', '{:.6e}'.format)
    print(df_results['rel_diff'].describe())


if __name__ == '__main__':
    main()
