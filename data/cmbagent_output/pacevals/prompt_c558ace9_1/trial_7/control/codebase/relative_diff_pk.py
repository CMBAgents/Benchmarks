# filename: codebase/relative_diff_pk.py
import numpy as np
import pandas as pd
import camb
from camb import model
import matplotlib.pyplot as plt
import os
import time


def calculate_power_spectra_and_diff():
    """
    Calculates the linear matter power spectrum for normal and inverted neutrino hierarchies,
    computes their relative difference, generates a plot, and saves the results to a CSV file.
    """
    # Cosmological parameters
    H0_param = 67.5  # Hubble constant in km/s/Mpc
    ombh2_param = 0.022  # Baryon density
    omch2_param = 0.122  # Cold dark matter density
    mnu_total_param = 0.11  # Sum of neutrino masses in eV
    As_param = 2.0e-9  # Scalar amplitude
    ns_param = 0.965  # Scalar spectral index
    omk_param = 0.0  # Curvature density (flat universe)
    
    # k-range parameters
    kh_min_param = 1e-4  # Minimum k*h in Mpc^-1
    kh_max_param = 2.0  # Maximum k*h in Mpc^-1
    n_kpoints_param = 200  # Number of k points
    z_target_param = 0.0  # Redshift

    # Neutrino mass fractions (sum to 1)
    # These fractions are derived from typical mass splittings for a sum of 0.11 eV
    # For Normal Hierarchy (m1 < m2 << m3):
    # Using del_m_21_sq = 7.5e-5 eV^2, del_m_31_sq = 2.5e-3 eV^2
    # m1 ~ 0.0085 eV, m2 ~ 0.0123 eV, m3 ~ 0.0508 eV for sum = 0.0716 (example, not 0.11)
    # For sum = 0.11 eV:
    # NH: m1=0.0097, m2=0.0137, m3=0.0866. Fractions: m1/0.11, m2/0.11, m3/0.11
    # These are approximate, precise calculation depends on solving system of equations.
    # Using fractions from context:
    nu_mass_fractions_NH = [0.0097/mnu_total_param, 0.0137/mnu_total_param, 0.0866/mnu_total_param] 
    # Re-normalizing to ensure sum is 1, if the above are not precise enough for CAMB
    sum_nh_masses_approx = 0.0097 + 0.0137 + 0.0866  # This is approx 0.11
    nu_mass_fractions_NH = [0.0097/sum_nh_masses_approx, 0.0137/sum_nh_masses_approx, 0.0866/sum_nh_masses_approx]


    # For Inverted Hierarchy (m3 << m1 < m2):
    # Using del_m_21_sq = 7.5e-5 eV^2, del_m_23_sq = -2.4e-3 eV^2 (so del_m_13_sq ~ 2.4e-3)
    # m3 ~ 0.0085, m1 ~ 0.0498, m2 ~ 0.0506 for sum = 0.1089 (example)
    # For sum = 0.11 eV:
    # IH: m3=0.0049, m1=0.0521, m2=0.0530. Fractions: m3/0.11, m1/0.11, m2/0.11
    # Order for CAMB is m1, m2, m3 by convention, so for IH it's [m_lightest, m_middle, m_heaviest]
    # if CAMB expects physical masses in order.
    # However, nu_mass_fractions usually corresponds to the three mass eigenstates.
    # Let's use the fractions provided in the problem context which are already ordered.
    nu_mass_fractions_NH_context = [0.23558, 0.24862, 0.51580]  # Sums to 1.0
    nu_mass_fractions_IH_context = [0.04448, 0.47392, 0.48097]  # Sums to 1.0

    # Using context-provided fractions as they are likely more accurate or standard for this sum.
    nu_mass_fractions_NH = nu_mass_fractions_NH_context
    nu_mass_fractions_IH = nu_mass_fractions_IH_context


    print("CAMB version: " + str(camb.__version__))

    # Create data directory if it doesn't exist
    database_path = "data"
    if not os.path.exists(database_path):
        os.makedirs(database_path)
        print("Created directory: " + database_path)

    def get_pk(mnu_sum, nu_fractions, num_massive_neutrinos_param=3):
        """
        Computes the linear matter power spectrum for a given set of neutrino parameters.

        Args:
            mnu_sum (float): Total sum of neutrino masses in eV.
            nu_fractions (list): List of mass fractions for each neutrino eigenstate.
            num_massive_neutrinos_param (int): Number of massive neutrino species.

        Returns:
            tuple: (kh, pk)
                kh (numpy.ndarray): Array of wavenumbers k*h (Mpc^-1).
                pk (numpy.ndarray): Array of power spectrum P(k) ((Mpc/h)^3).
        """
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0_param, ombh2=ombh2_param, omch2=omch2_param, mnu=mnu_sum, omk=omk_param)
        pars.set_massivenu_params(nu_mass_fractions=nu_fractions, num_massive_neutrinos=num_massive_neutrinos_param)
        pars.InitPower.set_params(As=As_param, ns=ns_param)
        
        # We want P(k) at z=0. kmax in set_matter_power should be at least kh_max_param.
        pars.set_matter_power(redshifts=[z_target_param], kmax=kh_max_param)
        pars.NonLinear = model.NonLinear_none  # Ensure linear power spectrum
        
        results = camb.get_results(pars)
        
        kh_vals, zs_vals, pk_vals = results.get_matter_power_spectrum(
            minkh=kh_min_param, maxkh=kh_max_param, npoints=n_kpoints_param,
            nonlinear=False, var1='delta_tot', var2='delta_tot',
            hubble_units=True, k_hunit=True
        )
        # pk_vals is 2D (z, k), we need pk_vals[0] for z=0
        return kh_vals, pk_vals[0]

    # Get power spectrum for Normal Hierarchy
    print("Calculating P(k) for Normal Hierarchy...")
    kh_nh, pk_nh = get_pk(mnu_total_param, nu_mass_fractions_NH)
    print("P(k) for Normal Hierarchy calculated.")

    # Get power spectrum for Inverted Hierarchy
    print("Calculating P(k) for Inverted Hierarchy...")
    kh_ih, pk_ih = get_pk(mnu_total_param, nu_mass_fractions_IH)
    print("P(k) for Inverted Hierarchy calculated.")

    # Ensure k-values are the same (they should be by construction)
    if not np.allclose(kh_nh, kh_ih):
        print("Warning: k-values from NH and IH calculations differ slightly. Using NH k-values.")
        # This case should ideally not happen if minkh, maxkh, npoints are identical.
    
    kh_common = kh_nh  # Use one of them as they should be identical

    # Calculate relative difference
    # Avoid division by zero if pk_nh can be zero, though unlikely for P(k) > 0
    # Add a small epsilon or check for pk_nh == 0 if necessary.
    # For P(k), values are typically positive.
    rel_diff = (pk_ih / pk_nh) - 1.0

    # --- Plotting ---
    plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(kh_common, rel_diff)
    ax.set_xscale('log')
    ax.set_xlabel("Wavenumber, k (h/Mpc)")
    ax.set_ylabel("Relative Difference ((P_IH / P_NH) - 1)")
    ax.set_title("Relative Difference in Linear Matter Power Spectrum (z=" + str(z_target_param) + ")")
    ax.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plot_filename = os.path.join(database_path, "rel_diff_pk_plot_1_" + timestamp + ".png")
    plt.savefig(plot_filename, dpi=300)
    print("Plot saved to: " + plot_filename)
    print("Plot description: Shows the relative difference in the linear matter power spectrum at z=" + str(z_target_param) + " between inverted and normal neutrino hierarchies as a function of wavenumber k.")

    # --- Saving to CSV ---
    df_results = pd.DataFrame({
        'k': kh_common,  # k in h/Mpc
        'rel_diff': rel_diff  # Dimensionless
    })
    
    csv_filename = os.path.join(database_path, "result.csv")
    df_results.to_csv(csv_filename, index=False)
    print("Results saved to CSV: " + csv_filename)

    # Print some results to console
    print("\nSample of calculated results:")
    print("k (h/Mpc) | Relative Difference")
    print("------------------------------------")
    for i in range(0, len(kh_common), len(kh_common) // 10):  # Print 10 samples
        print(str(round(kh_common[i], 5)) + "      | " + str(round(rel_diff[i], 5)))
    
    print("\nMaximum relative difference: " + str(round(np.max(rel_diff), 5)))
    print("Minimum relative difference: " + str(round(np.min(rel_diff), 5)))
    
    # Sanity check: print first few k and P(k) values for NH and IH
    print("\nFirst 3 P(k) values for Normal Hierarchy:")
    for i in range(3):
        print("k=" + str(round(kh_nh[i], 6)) + " h/Mpc, P(k)_NH=" + str(round(pk_nh[i], 2)) + " (Mpc/h)^3")

    print("\nFirst 3 P(k) values for Inverted Hierarchy:")
    for i in range(3):
        print("k=" + str(round(kh_ih[i], 6)) + " h/Mpc, P(k)_IH=" + str(round(pk_ih[i], 2)) + " (Mpc/h)^3")


if __name__ == '__main__':
    calculate_power_spectra_and_diff()
