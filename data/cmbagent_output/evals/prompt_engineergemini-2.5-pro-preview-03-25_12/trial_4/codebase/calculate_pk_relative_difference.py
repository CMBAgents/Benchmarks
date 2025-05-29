# filename: codebase/calculate_pk_relative_difference.py
import numpy as np
import pandas as pd
import camb
import os
import io


def get_pk_for_hierarchy(hierarchy_type_str, H0_val, ombh2_val, omch2_val, sum_mnu_val, As_val, ns_val, omk_val, redshift_val, minkh_val, maxkh_val, npoints_val):
    r"""
    Calculates the linear matter power spectrum for a given neutrino hierarchy.

    Parameters
    ----------
    hierarchy_type_str : str
        Neutrino hierarchy type: 'normal' or 'inverted'.
    H0_val : float
        Hubble constant (km/s/Mpc).
    ombh2_val : float
        Baryon density Omega_b * h^2.
    omch2_val : float
        Cold dark matter density Omega_c * h^2.
    sum_mnu_val : float
        Sum of neutrino masses (eV).
    As_val : float
        Scalar amplitude.
    ns_val : float
        Scalar spectral index.
    omk_val : float
        Curvature density Omega_k.
    redshift_val : float
        Redshift at which to calculate power spectrum.
    minkh_val : float
        Minimum k (h/Mpc) for the output power spectrum.
    maxkh_val : float
        Maximum k (h/Mpc) for the output power spectrum.
    npoints_val : int
        Number of k points for the output power spectrum.

    Returns
    -------
    k_vals_out : np.ndarray
        Array of k values (units: h/Mpc).
    pk_vals_out : np.ndarray
        Array of P(k) values (units: (Mpc/h)^3).
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0_val, ombh2=ombh2_val, omch2=omch2_val, mnu=sum_mnu_val, omk=omk_val)
    
    pars.InitPower.As = As_val
    pars.InitPower.ns = ns_val
    
    pars.set_matter_power(redshifts=[redshift_val], kmax=maxkh_val * 1.2) 
    pars.NonLinear = camb.model.NonLinear_none

    pars.num_massive_neutrinos = 3
    if hierarchy_type_str == 'normal':
        pars.nu_mass_eigenstates = 1
    elif hierarchy_type_str == 'inverted':
        pars.nu_mass_eigenstates = 2
    else:
        raise ValueError("hierarchy_type_str must be 'normal' or 'inverted'")

    results = camb.get_results(pars)
    
    k_arr, _, pk_arr = results.get_matter_power_spectrum(
        minkh=minkh_val, maxkh=maxkh_val, npoints=npoints_val, nonlinear=False
    )
    return k_arr, pk_arr[0]


def main():
    r"""
    Main function to perform calculations and save results.
    """
    # Define cosmological parameters
    H0 = 67.5  # Hubble constant (km/s/Mpc)
    ombh2 = 0.022  # Baryon density Omega_b * h^2
    omch2 = 0.122  # Cold dark matter density Omega_c * h^2
    sum_mnu = 0.11  # Sum of neutrino masses (eV)
    As_val = 2.0e-9  # Scalar amplitude
    ns_val = 0.965  # Scalar spectral index
    omk = 0.0  # Curvature density Omega_k (0 for flat)
    h_param = H0 / 100.0  # Dimensionless Hubble parameter

    # k-range for power spectrum
    # The problem states: "200 evenly spaced k values in the range 10^-4 < k h < 2 (Mpc^-1)"
    # Let k_phys be the physical wavenumber in Mpc^-1. "k h" is k_phys * h_param.
    # So, 1e-4 < k_phys * h_param < 2.
    # The k values for CAMB/CSV are k_output = k_phys / h_param (units h/Mpc).
    # k_output = (k_phys * h_param) / (h_param^2).
    # Thus, k_output range is [1e-4 / h_param**2, 2.0 / h_param**2].
    minkh_spec_range = 1e-4
    maxkh_spec_range = 2.0
    minkh = minkh_spec_range / (h_param**2)
    maxkh = maxkh_spec_range / (h_param**2)
    npoints = 200

    # Redshift
    redshift = 0.0

    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created directory: " + output_dir)

    print("Calculating P(k) for normal hierarchy...")
    print("Using k range (h/Mpc): " + str(minkh) + " to " + str(maxkh))
    k_values, pk_normal = get_pk_for_hierarchy('normal', H0, ombh2, omch2, sum_mnu, As_val, ns_val, omk, redshift, minkh, maxkh, npoints)
    print("P(k) for normal hierarchy calculated.")

    print("Calculating P(k) for inverted hierarchy...")
    k_values_IH, pk_inverted = get_pk_for_hierarchy('inverted', H0, ombh2, omch2, sum_mnu, As_val, ns_val, omk, redshift, minkh, maxkh, npoints)
    print("P(k) for inverted hierarchy calculated.")

    if not np.allclose(k_values, k_values_IH):
        print("Warning: k-values from normal and inverted hierarchy calculations differ slightly.")
    
    if np.any(pk_normal == 0):
        print("Error: pk_normal contains zero values. Cannot compute relative difference.")
        # For this task, we assume pk_normal will not be zero.
        # If it could be, robust error handling (e.g. raising ValueError) would be needed.
    
    rel_diff = (pk_inverted / pk_normal) - 1.0

    df_results = pd.DataFrame({'k': k_values, 'rel_diff': rel_diff})

    output_filename = os.path.join(output_dir, 'result.csv')
    df_results.to_csv(output_filename, index=False, float_format='%.8e')
    print("Results saved to " + output_filename)

    print("\nSummary of results (first 5 rows):")
    print(df_results.head().to_string())
    print("\nSummary of results (last 5 rows):")
    print(df_results.tail().to_string())
    
    buffer = io.StringIO()
    df_results.info(buf=buffer)  # Get full info
    info_str = buffer.getvalue()
    print("\nDataFrame Info:")
    print(info_str)


if __name__ == '__main__':
    main()