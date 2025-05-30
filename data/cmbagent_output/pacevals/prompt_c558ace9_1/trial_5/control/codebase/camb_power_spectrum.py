# filename: codebase/camb_power_spectrum.py
import camb
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import matplotlib


def calculate_matter_power_spectra():
    """
    Calculates the linear matter power spectrum for normal and inverted
    neutrino hierarchies using CAMB, computes the relative difference,
    saves the results to a CSV file, and visualizes the relative difference.

    The cosmological parameters are:
    H0 = 67.5 km/s/Mpc
    ombh2 = 0.022
    omch2 = 0.122
    mnu = 0.11 eV
    As = 2e-9
    ns = 0.965
    omk = 0 (flat universe)

    The output k-values for the CSV and plot are k_phys * h, ranging from 1e-4 to 2.0 (h/Mpc).
    The power spectrum P(k) used for the difference is P(k_phys) in units of (Mpc/h)^3.
    The relative difference is (P_inverted(k_phys) / P_normal(k_phys) - 1).
    Results are saved to data/result.csv.
    A plot of the relative difference is saved to data/relative_power_spectrum_difference_1_<timestamp>.png.
    """
    matplotlib.rcParams['text.usetex'] = False

    # Cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Physical baryon density
    omch2 = 0.122  # Physical cold dark matter density
    sum_mnu = 0.11  # Sum of neutrino masses in eV
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    omk = 0.0  # Curvature density parameter (0 for flat)

    h = H0 / 100.0  # Dimensionless Hubble parameter

    # k-range for the output CSV file and plot (k_csv = k_phys * h)
    # k_csv has units of h/Mpc.
    # The problem states "10^-4 < k h < 2 (Mpc^-1)", so k_csv_max = 2.0.
    k_min_csv = 1e-4  # min k_phys * h
    k_max_csv = 2.0   # max k_phys * h
    npoints = 200   # Number of k points

    # Derived parameters for CAMB's get_matter_power_spectrum
    # We want k_csv = k_camb_output * h, where k_camb_output = k_phys.
    # So k_phys should be in [k_min_csv/h, k_max_csv/h].
    # For get_matter_power_spectrum with k_hunit=True, minkh and maxkh are k_phys/h.
    # So, minkh_param = (k_min_csv/h)/h = k_min_csv / (h**2)
    # And maxkh_param = (k_max_csv/h)/h = k_max_csv / (h**2)
    minkh_param_camb = k_min_csv / (h**2)
    maxkh_param_camb = k_max_csv / (h**2)

    # kmax for CAMB's set_matter_power (physical k_max in Mpc^-1)
    # This should be the max k_phys we are interested in.
    kmax_internal_camb = k_max_csv / h

    print("Cosmological Parameters:")
    print("H0: " + str(H0) + " km/s/Mpc")
    print("h: " + str(h))
    print("ombh2: " + str(ombh2))
    print("omch2: " + str(omch2))
    print("sum_mnu: " + str(sum_mnu) + " eV")
    print("As: " + str(As))
    print("ns: " + str(ns))
    print("omk: " + str(omk))
    print("\nCAMB input parameters for get_matter_power_spectrum:")
    print("minkh (k_phys/h for grid): " + str(minkh_param_camb))
    print("maxkh (k_phys/h for grid): " + str(maxkh_param_camb))
    print("kmax for set_matter_power (k_phys): " + str(kmax_internal_camb) + " Mpc^-1")
    print("npoints: " + str(npoints))


    def get_pk_for_hierarchy(hierarchy_model):
        """
        Computes the linear matter power spectrum for a given neutrino hierarchy.

        Args:
            hierarchy_model (str): 'normal' or 'inverted'.

        Returns:
            tuple: (k_values_for_csv, power_spectrum_values)
                   k_values_for_csv are k_phys * h (units: h/Mpc)
                   power_spectrum_values are P(k_phys) (units: (Mpc/h)^3)
        """
        pars = camb.CAMBparams()
        try:
            pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omk,
                               mnu=sum_mnu, neutrino_hierarchy=hierarchy_model)
            pars.InitPower.set_params(As=As, ns=ns)
            pars.set_matter_power(redshifts=[0.], kmax=kmax_internal_camb)
            pars.NonLinear = camb.model.NonLinear_none

            results = camb.get_results(pars)
            
            # hubble_units=True: Pk in (Mpc/h)^3, k_camb_output in Mpc^-1 (k_phys)
            # k_hunit=True: minkh, maxkh are k_phys/h
            # So, k_camb_output (returned k) = (k_phys/h)_grid * h = k_phys
            # And pk_camb_output (returned Pk) = P(k_phys)
            # log_interp=True is default, giving log-spaced k points.
            k_camb_output, _, pk_camb_output = results.get_matter_power_spectrum(
                minkh=minkh_param_camb, maxkh=maxkh_param_camb, npoints=npoints,
                var1='delta_tot', var2='delta_tot',
                hubble_units=True, k_hunit=True, nonlinear=False
            )
            # k_values_for_csv should be k_phys * h
            k_values_for_csv = k_camb_output * h
            return k_values_for_csv, pk_camb_output[0]  # pk_camb is 2D (z,k), we take z=0
        except Exception as e:
            print("Error calculating power spectrum for " + hierarchy_model + " hierarchy: " + str(e))
            raise

    try:
        print("\nCalculating for Normal Hierarchy...")
        k_normal_csv, pk_normal = get_pk_for_hierarchy('normal')
        
        print("Calculating for Inverted Hierarchy...")
        k_inverted_csv, pk_inverted = get_pk_for_hierarchy('inverted')

        if not np.allclose(k_normal_csv, k_inverted_csv):
            print("Warning: k-arrays from normal and inverted hierarchy calculations differ significantly.")
        
        k_values_output = k_normal_csv  # k_phys * h in h/Mpc

        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = (pk_inverted / pk_normal) - 1
            rel_diff[~np.isfinite(rel_diff)] = 0 

        df_results = pd.DataFrame({
            'k': k_values_output,  # k in h/Mpc
            'rel_diff': rel_diff  # Relative difference
        })

        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print("Created directory: " + data_dir)

        # Prepare metadata for CSV
        metadata_lines = [
            "# Relative difference in linear matter power spectrum P(k)",
            "# P(k) calculated using CAMB",
            "# Redshift z=0",
            "# Cosmological Parameters:",
            "# H0: " + str(H0) + " km/s/Mpc",
            "# ombh2: " + str(ombh2),
            "# omch2: " + str(omch2),
            "# sum_mnu: " + str(sum_mnu) + " eV",
            "# As: " + str(As),
            "# ns: " + str(ns),
            "# omk: " + str(omk) + " (flat universe)",
            "# Neutrino hierarchy for P_inverted: inverted",
            "# Neutrino hierarchy for P_normal: normal",
            "# k column: Wavenumber k_phys * h (units: h/Mpc)",
            "# rel_diff column: Relative difference (P_inverted(k_phys) / P_normal(k_phys)) - 1",
            "# P(k_phys) values used for difference are in (Mpc/h)^3",
        ]
        csv_metadata_header = "\n".join(metadata_lines) + "\n"

        csv_filename = os.path.join(data_dir, "result.csv")
        
        # Write metadata and DataFrame to CSV
        with open(csv_filename, 'w') as f:
            f.write(csv_metadata_header)
            df_results.to_csv(f, index=False, lineterminator='\n')
            
        print("\nResults saved to: " + csv_filename)
        print("\nFirst 5 rows of the results (from " + csv_filename + "):")
        print(df_results.head().to_string())
        
        # Plotting
        print("\nGenerating plot...")
        plt.figure(figsize=(10, 6))
        plt.plot(k_values_output, rel_diff)
        plt.xscale('log')
        plt.xlabel("Wavenumber k (h/Mpc)")
        plt.ylabel("Relative Difference ((P_inv / P_norm) - 1)")
        plt.title("Relative Difference in Matter Power Spectrum (Inverted vs Normal Hierarchy)")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.tight_layout()

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plot_filename = os.path.join(data_dir, "relative_power_spectrum_difference_1_" + timestamp + ".png")
        plt.savefig(plot_filename, dpi=300)
        plt.close()  # Close the figure to free memory
        print("Plot saved to: " + plot_filename)
        print("Description: Plot of relative difference in matter power spectrum vs. wavenumber k (log scale for k).")


        print("\nSummary statistics for relative difference:")
        print("Min rel_diff: " + str(np.min(rel_diff)))
        print("Max rel_diff: " + str(np.max(rel_diff)))
        print("Mean rel_diff: " + str(np.mean(rel_diff)))
        
        print("\nOutput k-range (h/Mpc):")
        print("Min k: " + str(np.min(k_values_output)))
        print("Max k: " + str(np.max(k_values_output)))
        print("Number of k points: " + str(len(k_values_output)))


    except Exception as e:
        print("An error occurred during the calculation or plotting: " + str(e))
        # Optionally, re-raise the exception if it should halt execution further up
        # raise


if __name__ == '__main__':
    calculate_matter_power_spectra()