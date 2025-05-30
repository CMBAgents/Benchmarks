# filename: codebase/camb_linear_power_spectrum.py
import os
import camb
from camb import model
import numpy as np
import matplotlib.pyplot as plt
import time


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
    Main function to calculate, validate, and plot power spectra, and compute relative difference.
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
    Pk_NH = get_power_spectrum(pars_NH, kh_input_values, z_pk)  # P(k/h) in (Mpc/h)^3
    
    print("Computing P(k) for Inverted Hierarchy...")
    Pk_IH = get_power_spectrum(pars_IH, kh_input_values, z_pk)  # P(k/h) in (Mpc/h)^3

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
    ax.set_title('Linear Matter Power Spectrum (z=0)')
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
    print("Description: Log-log plot of the linear matter power spectrum P(k/h) at redshift z=0 for normal and inverted neutrino hierarchies. k/h is in h/Mpc, P(k/h) is in (Mpc/h)^3.")

    # Calculate relative difference
    # (P(k)_inverted / P(k)_normal - 1)
    # Avoid division by zero if Pk_NH can be zero, though unlikely for P(k) > 0.
    if np.any(Pk_NH == 0):
        print("Warning: Zero values found in Pk_NH, relative difference might be problematic.")
    
    rel_diff = (Pk_IH / Pk_NH) - 1.0
    
    print("\nRelative difference (P_IH / P_NH - 1):")
    print("Min relative difference: " + str(np.min(rel_diff)))
    print("Max relative difference: " + str(np.max(rel_diff)))
    
    # Data for CSV (k, rel_diff) will be prepared for the next step.
    # For now, we have kh_input_values and rel_diff.


if __name__ == '__main__':
    main()
