# filename: codebase/calculate_power_spectra.py
import os
import camb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Set Matplotlib parameters
plt.rcParams['text.usetex'] = False  # Ensure LaTeX rendering is off
plt.rcParams['figure.dpi'] = 300  # Set default DPI for figures


def calculate_power_spectra_and_diff():
    """
    Calculates the linear matter power spectrum for normal and inverted neutrino hierarchies,
    computes their relative difference, saves the results to a CSV file, and plots the difference.

    Cosmological parameters are fixed as per the problem description.
    k values are in h/Mpc.
    Power spectra P(k) are in (Mpc/h)^3.
    """
    try:
        # Cosmological parameters
        H0_val = 67.5  # Hubble constant in km/s/Mpc
        ombh2_val = 0.022  # Physical baryon density Omega_b * h^2
        omch2_val = 0.122  # Physical cold dark matter density Omega_c * h^2
        mnu_sum_val = 0.11  # Sum of neutrino masses in eV
        As_val = 2.0e-9  # Scalar amplitude
        ns_val = 0.965  # Scalar spectral index
        omk_val = 0.0  # Curvature density Omega_k (0 for flat)
        z_pk = 0.0  # Redshift for power spectrum calculation

        # k-range for P(k)
        # k_h_values are k_physical / h_param, commonly denoted as k (h/Mpc)
        k_h_min = 1e-4  # min k (h/Mpc)
        k_h_max = 2.0  # max k (h/Mpc)
        n_points = 200  # Number of k points
        # k_h_values are the wavenumbers in h/Mpc for which P(k) will be computed
        k_h_values = np.linspace(k_h_min, k_h_max, n_points)  # Array of k values in h/Mpc

        # h_param is the dimensionless Hubble parameter: H0 / 100
        h_param = H0_val / 100.0
        # kmax_phys_calc is the maximum physical k (in Mpc^-1) for CAMB's internal calculations.
        # It should be slightly larger than the maximum physical k we need for interpolation.
        # Max physical k needed for interpolation = k_h_max * h_param
        kmax_phys_calc = k_h_max * h_param * 1.05  # Physical k in Mpc^-1

        print("Setting up CAMB parameters...")
        print("H0: " + str(H0_val) + " km/s/Mpc")
        print("Omega_b h^2: " + str(ombh2_val))
        print("Omega_c h^2: " + str(omch2_val))
        print("Sum of neutrino masses (Sigma m_nu): " + str(mnu_sum_val) + " eV")
        print("Scalar amplitude (A_s): " + str(As_val))
        print("Scalar spectral index (n_s): " + str(ns_val))
        print("Omega_k: " + str(omk_val))
        print("Redshift for P(k): " + str(z_pk))
        print("k range (for k in h/Mpc): " + str(k_h_min) + " to " + str(k_h_max) + " with " + str(n_points) + " points.")

        def get_pk(neutrino_hierarchy_str):
            """
            Computes the linear matter power spectrum P(k) for a given neutrino hierarchy.

            :param neutrino_hierarchy_str: A string, either 'normal' or 'inverted', specifying the
                                           neutrino mass hierarchy.
            :return: A numpy array of P(k) values. Units are (Mpc/h)^3.
                     The corresponding k values are k_h_values (in h/Mpc).
            """
            pars = camb.CAMBparams()
            pars.set_cosmology(
                H0=H0_val,
                ombh2=ombh2_val,
                omch2=omch2_val,
                mnu=mnu_sum_val,
                omk=omk_val,
                num_massive_neutrinos=3,  # Standard model has 3 massive neutrino species
                neutrino_hierarchy=neutrino_hierarchy_str
            )
            
            pars.InitPower.set_params(As=As_val, ns=ns_val)  # Set primordial power spectrum parameters
            
            # Ensure linear power spectrum is computed
            pars.NonLinear = camb.model.NonLinear_none 
            
            # Set redshifts and kmax for CAMB's internal calculation grid
            # kmax here is in physical units (Mpc^-1)
            pars.set_matter_power(redshifts=[z_pk], kmax=kmax_phys_calc)
            
            print("Running CAMB for " + neutrino_hierarchy_str + " hierarchy...")
            results = camb.get_results(pars)  # This runs the CAMB calculation
            
            # Get the linear matter power spectrum interpolator object
            # k_hunit=True means input k values to the interpolator are k/h (units h/Mpc)
            # hubble_units=True means output P(k) is in (Mpc/h)^3
            # kmax for interpolator is max k/h it should cover
            PK_interpolator = results.get_matter_power_interpolator(
                nonlinear=False,  # Explicitly request linear P(k)
                var1='delta_tot',  # Matter fluctuations
                var2='delta_tot',  # Matter fluctuations
                hubble_units=True, 
                k_hunit=True, 
                kmax=k_h_max 
            )
            
            # Evaluate P(k) at z=z_pk for the specified k_h_values
            pk_values_output = PK_interpolator.P(z_pk, k_h_values)  # P(k) in (Mpc/h)^3
            return pk_values_output

        # Compute P(k) for normal hierarchy
        pk_normal = get_pk('normal')  # Array of P(k) values for normal hierarchy
        
        # Compute P(k) for inverted hierarchy
        pk_inverted = get_pk('inverted')  # Array of P(k) values for inverted hierarchy

        print("CAMB calculations complete.")

        # Calculate relative difference: (P(k)_inverted / P(k)_normal - 1)
        # Using np.divide for safe division. If pk_normal is zero at some point where pk_inverted is non-zero, 
        # rel_diff will be inf. If both are zero, it will be nan. Then -1.0 is applied.
        # For P(k) in this range, pk_normal should not be zero.
        # The `out` argument specifies what to fill in elements where `where` is False (i.e., pk_normal is 0).
        # So, if pk_normal is 0, that element of rel_diff becomes (0.0 - 1.0) = -1.0.
        rel_diff = np.divide(pk_inverted, pk_normal, out=np.zeros_like(pk_inverted, dtype=float), where=pk_normal != 0) - 1.0
        
        # Check if any division by zero occurred (where pk_normal was 0 and pk_inverted was non-zero, leading to inf)
        # or where both were zero (leading to nan).
        if np.any(pk_normal == 0):
             print("Warning: Division by zero encountered in relative difference calculation where P(k)_normal was zero. " +
                   "Affected rel_diff values were set to -1.0 if P(k)_inverted was also zero, or potentially inf/nan otherwise if not handled by `out`.")
        if np.any(np.isinf(rel_diff)) or np.any(np.isnan(rel_diff)):
            print("Warning: Relative difference contains inf or NaN values. This might indicate issues with P(k) values (e.g., zeros).")

        
        print("Relative difference calculation complete.")

        # Find maximum absolute relative difference and corresponding k
        # We filter out NaNs and Infs before finding max, if any occurred.
        valid_rel_diff = rel_diff[np.isfinite(rel_diff)]
        if len(valid_rel_diff) > 0:
            max_abs_rel_diff_idx_in_valid = np.argmax(np.abs(valid_rel_diff))
            # To get the original index in rel_diff and k_h_values:
            # This is a bit complex if we want the exact original k. Simpler to report on valid values.
            max_abs_rel_diff_val = valid_rel_diff[max_abs_rel_diff_idx_in_valid]
            
            # Find corresponding k value (this assumes k_h_values maps directly to finite rel_diff values)
            # A more robust way:
            finite_indices = np.where(np.isfinite(rel_diff))[0]
            original_idx_for_max = finite_indices[np.argmax(np.abs(rel_diff[finite_indices]))]
            k_at_max_abs_rel_diff = k_h_values[original_idx_for_max]  # k (h/Mpc)
            max_abs_rel_diff_val_check = rel_diff[original_idx_for_max]


            print("Maximum absolute relative difference (among finite values): " + str(max_abs_rel_diff_val_check))
            print("Occurs at k = " + str(k_at_max_abs_rel_diff) + " h/Mpc")
        else:
            print("Warning: All relative difference values are non-finite (NaN or Inf). Cannot determine maximum.")
        
        # Create data directory if it doesn't exist
        data_dir = "data"  # Save files in data/ subdirectory
        os.makedirs(data_dir, exist_ok=True)

        # Save results to CSV file
        # Columns: k (wavenumber in h/Mpc), rel_diff (relative difference)
        csv_filename = os.path.join(data_dir, "result.csv")
        results_df = pd.DataFrame({
            'k': k_h_values,  # k in h/Mpc
            'rel_diff': rel_diff  # Dimensionless relative difference
        })
        results_df.to_csv(csv_filename, index=False, float_format='%.8e')  # Use scientific notation for precision
        print("Results saved to " + csv_filename)

        # Create and save plot of the relative difference
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        plot_filename = os.path.join(data_dir, "relative_difference_pk_1_" + timestamp + ".png")
        
        fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axes
        ax.semilogx(k_h_values, rel_diff)  # Plot k on a log scale
        ax.set_xlabel("k (h/Mpc)")  # X-axis label with units
        ax.set_ylabel("Relative Difference ((P_inv / P_nor) - 1)")  # Y-axis label (dimensionless)
        ax.set_title("Relative Difference in Linear P(k) (Inverted vs Normal Hierarchy)")  # Plot title
        ax.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)  # Add grid lines
        fig.tight_layout()  # Adjust layout to prevent labels from overlapping
        plt.savefig(plot_filename)  # Save plot to file
        plt.close(fig)  # Close the figure to free memory
        
        print("Plot saved to " + plot_filename)
        plot_description = ("This plot shows the relative difference in the linear matter power spectrum P(k) " +
                            "at z=0 between inverted and normal neutrino hierarchies. The x-axis is the " +
                            "wavenumber k in h/Mpc (logarithmic scale), and the y-axis is the " +
                            "relative difference (P_inverted / P_normal - 1).")
        print("Plot description: " + plot_description)

    except ImportError:
        print("Error: CAMB package not found. Please ensure CAMB is installed.")
        raise
    except Exception as e:
        print("An error occurred during the execution:")
        import traceback
        print(traceback.format_exc())
        raise


if __name__ == '__main__':
    calculate_power_spectra_and_diff()
