# filename: codebase/linear_matter_power_spectrum.py
import os
import camb
from camb import model
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

def configure_camb_params(H0, ombh2, omch2, mnu_sum, As, ns, omk, hierarchy_model):
    """
    Configures and returns CAMBparams object for given cosmological parameters and neutrino hierarchy.

    Args:
        H0 (float): Hubble constant in km/s/Mpc.
        ombh2 (float): Baryon density parameter.
        omch2 (float): Cold dark matter density parameter.
        mnu_sum (float): Sum of neutrino masses in eV.
        As (float): Scalar amplitude.
        ns (float): Scalar spectral index.
        omk (float): Curvature density parameter.
        hierarchy_model (camb.model.NuMassHierarchy): Neutrino mass hierarchy model 
                         (e.g., model.NuMassNormalHierarchical, model.NuMassInvertedHierarchical).

    Returns:
        camb.CAMBparams: Configured CAMB parameters object.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu_sum, omk=omk, num_massive_neutrinos=3)
    pars.nu_mass_model = hierarchy_model
    pars.InitPower.set_params(As=As, ns=ns)
    # kmax is in Mpc^-1. Our k/h values go up to 2 h/Mpc.
    # Max k is 2 * (H0/100) = 2 * 0.675 = 1.35 Mpc^-1.
    # So kmax=2.0 Mpc^-1 is sufficient.
    pars.set_matter_power(redshifts=[0.0], kmax=2.0)
    return pars

def get_power_spectrum(pars, kh_values, redshift):
    """
    Computes the linear matter power spectrum for given CAMB parameters and k/h values.

    Args:
        pars (camb.CAMBparams): Configured CAMB parameters object.
        kh_values (np.ndarray): Array of k/h values (units: h/Mpc).
        redshift (float): Redshift at which to compute the power spectrum.

    Returns:
        np.ndarray: Linear matter power spectrum P(k/h) in (Mpc/h)^3.
    """
    results = camb.get_results(pars)
    # Get matter power spectrum interpolator
    # P(k/h) in (Mpc/h)^3 for k/h in h/Mpc
    PK = results.get_matter_power_interpolator(nonlinear=False, hubble_units=True, k_hunit=True)
    pk_values = PK.P(redshift, kh_values)
    return pk_values

def main():
    """
    Main function to calculate, validate, plot power spectra, compute relative difference, and save to CSV.
    """
    # Cosmological parameters
    H0_val = 67.5  # Hubble constant in km/s/Mpc
    ombh2_val = 0.022  # Baryon density
    omch2_val = 0.122  # Cold dark matter density
    mnu_sum_val = 0.11  # Neutrino mass sum in eV
    As_val = 2.0e-9  # Scalar amplitude
    ns_val = 0.965  # Scalar spectral index
    omk_val = 0.0  # Curvature density (flat cosmology)
    z_pk = 0.0  # Redshift for P(k)

    # k-range for P(k)
    # k/h values in h/Mpc
    kh_min = 1e-4
    kh_max = 2.0
    n_k_points = 200
    kh_input_values = np.linspace(kh_min, kh_max, n_k_points)  # units: h/Mpc

    print("Setting up CAMB for Normal Hierarchy...")
    pars_NH = configure_camb_params(H0_val, ombh2_val, omch2_val, mnu_sum_val, As_val, ns_val, omk_val, model.NuMassNormalHierarchical)
    
    print("Setting up CAMB for Inverted Hierarchy...")
    pars_IH = configure_camb_params(H0_val, ombh2_val, omch2_val, mnu_sum_val, As_val, ns_val, omk_val, model.NuMassInvertedHierarchical)

    print("Computing P(k) for Normal Hierarchy...")
    Pk_NH = get_power_spectrum(pars_NH, kh_input_values, z_pk) # P(k/h) in (Mpc/h)^3
    
    print("Computing P(k) for Inverted Hierarchy...")
    Pk_IH = get_power_spectrum(pars_IH, kh_input_values, z_pk) # P(k/h) in (Mpc/h)^3

    # Validation checks
    print("\nValidation Checks:")
    if np.isnan(Pk_NH).any():
        print("Error: NaN values found in P(k) for Normal Hierarchy.")
    else:
        print("P(k) for Normal Hierarchy seems valid (no NaNs).")
        print("Pk_NH[0] = " + str(Pk_NH[0]) + " (Mpc/h)^3, Pk_NH[-1] = " + str(Pk_NH[-1]) + " (Mpc/h)^3")


    if np.isnan(Pk_IH).any():
        print("Error: NaN values found in P(k) for Inverted Hierarchy.")
    else:
        print("P(k) for Inverted Hierarchy seems valid (no NaNs).")
        print("Pk_IH[0] = " + str(Pk_IH[0]) + " (Mpc/h)^3, Pk_IH[-1] = " + str(Pk_IH[-1]) + " (Mpc/h)^3")

    # Create visualization
    print("\nCreating plot of power spectra...")
    plt.rcParams['text.usetex'] = False  # Do not use LaTeX rendering
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(kh_input_values, Pk_NH, label='Normal Hierarchy')
    ax.loglog(kh_input_values, Pk_IH, label='Inverted Hierarchy', linestyle='--')
    
    ax.set_xlabel('k/h (h/Mpc)')
    ax.set_ylabel('P(k/h) ((Mpc/h)^3)')
    ax.set_title('Linear Matter Power Spectrum (z=' + str(z_pk) + ')')
    ax.legend()
    ax.grid(True, which="both", ls="-")
    
    plt.tight_layout()

    # Ensure data directory exists
    database_path = 'data/'
    os.makedirs(database_path, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d%H%M%S")
    plot_filename = database_path + 'linear_matter_power_spectra_1_' + timestamp + '.png'
    plt.savefig(plot_filename, dpi=300)
    print("Plot saved to: " + plot_filename)
    print("Description: Log-log plot of the linear matter power spectrum P(k/h) at redshift z=" + str(z_pk) + " for normal and inverted neutrino hierarchies. k/h is in h/Mpc, P(k/h) is in (Mpc/h)^3.")

    # Calculate relative difference
    # (P(k)_inverted / P(k)_normal - 1)
    if np.any(Pk_NH == 0):
        print("Warning: Zero values found in Pk_NH, relative difference might be problematic.")
        # Handle division by zero if Pk_NH can be zero. For P(k), values should be positive.
        # Replace with NaN or skip points where Pk_NH is zero.
        # For this problem, P(k) should not be zero for k > 0.
        # If it happens, it's an issue with CAMB or parameters.
        # We will proceed assuming Pk_NH is not zero.
        rel_diff = np.where(Pk_NH != 0, (Pk_IH / Pk_NH) - 1.0, np.nan)
        if np.isnan(rel_diff).any():
            print("NaNs generated in relative difference due to Pk_NH being zero at some k.")
    else:
        rel_diff = (Pk_IH / Pk_NH) - 1.0
    
    print("\nRelative difference (P_IH / P_NH - 1):")
    print("Min relative difference: " + str(np.nanmin(rel_diff)))
    print("Max relative difference: " + str(np.nanmax(rel_diff)))
    
    # Save results to CSV
    csv_filename = database_path + 'result.csv'
    
    metadata_header = ("# Relative difference in linear matter power spectrum P(k) at z=" + str(z_pk) + "\n" +
        "# between Inverted Hierarchy (IH) and Normal Hierarchy (NH) neutrino models.\n" +
        "# Relative difference = (P(k)_IH / P(k)_NH) - 1\n" +
        "# Cosmological Parameters:\n" +
        "# H0: " + str(H0_val) + " km/s/Mpc\n" +
        "# Omega_b h^2: " + str(ombh2_val) + "\n" +
        "# Omega_c h^2: " + str(omch2_val) + "\n" +
        "# Sum of neutrino masses (mnu): " + str(mnu_sum_val) + " eV\n" +
        "# Scalar amplitude (As): " + str(As_val) + "\n" +
        "# Scalar spectral index (ns): " + str(ns_val) + "\n" +
        "# Omega_k: " + str(omk_val) + " (Flat Universe)\n" +
        "# Number of massive neutrinos: 3\n" +
        "# k-range: " + str(kh_min) + " to " + str(kh_max) + " h/Mpc, " + str(n_k_points) + " points\n" +
        "# P(k) units: (Mpc/h)^3\n" +
        "# k units in CSV: h/Mpc\n"
    )

    df_results = pd.DataFrame({
        'k_h_Mpc_inv': kh_input_values,
        'relative_difference': rel_diff
    })

    try:
        with open(csv_filename, 'w') as f:
            f.write(metadata_header)
        df_results.to_csv(csv_filename, mode='a', index=False, header=True)
        print("\nResults saved to CSV: " + csv_filename)
        print("CSV file contains columns: 'k_h_Mpc_inv' (wavenumber in h/Mpc) and 'relative_difference'.")
        print("Metadata about parameters is included as comments in the CSV header.")
    except Exception as e:
        print("Error saving results to CSV: " + str(e))


if __name__ == '__main__':
    main()