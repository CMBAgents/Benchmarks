# filename: codebase/matter_power_spectrum.py
import os
import camb
import numpy as np
import pandas as pd


def calculate_matter_power_spectra():
    """
    Calculates the linear matter power spectrum P(k) at redshift z=0
    for normal and inverted neutrino hierarchies using CAMB,
    computes their relative difference, and saves the results to a CSV file.

    Cosmological Parameters:
        Hubble constant (H0): 67.5 km/s/Mpc
        Baryon density (ombh2): 0.022
        Cold dark matter density (omch2): 0.122
        Neutrino mass sum (mnu_total): 0.11 eV
        Scalar amplitude (As): 2.0e-9
        Scalar spectral index (ns): 0.965
        Omega_k (omk_flat): 0.0 (for flat cosmology)

    Power Spectrum Parameters:
        Min k*h (minkh_pk): 1e-4 h/Mpc
        Max k*h (maxkh_pk): 2.0 h/Mpc
        Number of k points (npoints_pk): 200
        Redshifts (redshifts_pk): [0.0]
    """

    # Cosmological parameters
    H0_param = 67.5  # Hubble constant in km/s/Mpc
    ombh2_param = 0.022  # Baryon density
    omch2_param = 0.122  # Cold dark matter density
    mnu_total_param = 0.11  # Sum of neutrino masses in eV
    As_param = 2.0e-9  # Scalar amplitude
    ns_param = 0.965  # Scalar spectral index
    omk_flat_param = 0.0  # Curvature density for flat cosmology

    # k-range and redshift for power spectrum
    minkh_pk_param = 1e-4  # Min k*h in h/Mpc
    maxkh_pk_param = 2.0  # Max k*h in h/Mpc
    npoints_pk_param = 200  # Number of k points
    redshifts_pk_param = [0.0]  # Redshift(s) for P(k)

    print("Setting up CAMB parameters...")

    def get_power_spectrum(neutrino_hierarchy_str):
        """
        Computes the linear matter power spectrum for a given neutrino hierarchy.

        Args:
            neutrino_hierarchy_str (str): 'normal' or 'inverted'.

        Returns:
            tuple: (kh_vals, pk_vals)
                   kh_vals (np.ndarray): Array of k/h values (h/Mpc).
                   pk_vals (np.ndarray): Array of P(k) values ((Mpc/h)^3).
                   Returns (None, None) if an error occurs.
        """
        try:
            pars = camb.CAMBparams()
            pars.set_cosmology(H0=H0_param, ombh2=ombh2_param, omch2=omch2_param,
                               mnu=mnu_total_param, omk=omk_flat_param,
                               num_massive_neutrinos=3,
                               neutrino_hierarchy=neutrino_hierarchy_str)
            pars.InitPower.set_params(As=As_param, ns=ns_param)
            # Set matter power spectrum parameters for linear P(k) at z=0
            # kmax here is the maximum k CAMB will calculate internally.
            pars.set_matter_power(redshifts=redshifts_pk_param, kmax=maxkh_pk_param,
                                  accurate_massive_neutrino_transfers=True)
            
            pars.NonLinear = camb.model.NonLinear_none  # Ensure linear power spectrum

            results = camb.get_results(pars)
            
            # Get linear matter power spectrum P(k/h) in (Mpc/h)^3
            # k/h in h/Mpc
            kh_vals, z_vals, pk_vals_all_z = results.get_matter_power_spectrum(
                minkh=minkh_pk_param, maxkh=maxkh_pk_param, npoints=npoints_pk_param,
                var1='delta_tot', var2='delta_tot',
                hubble_units=True, k_hunit=True,
                nonlinear=False
            )
            # We requested only one redshift, so pk_vals_all_z will have shape (npoints, 1)
            pk_vals = pk_vals_all_z[:, 0]
            return kh_vals, pk_vals
        except Exception as e:
            print("Error during CAMB computation for " + neutrino_hierarchy_str + " hierarchy: " + str(e))
            return None, None

    print("Calculating power spectrum for Normal Hierarchy...")
    kh_NH, Pk_normal = get_power_spectrum('normal')

    if kh_NH is None or Pk_normal is None:
        print("Failed to compute power spectrum for Normal Hierarchy. Exiting.")
        return

    if not (isinstance(Pk_normal, np.ndarray) and len(Pk_normal) == npoints_pk_param and np.all(Pk_normal > 0)):
        print("Warning: Pk_normal for Normal Hierarchy seems invalid (e.g. non-positive values or wrong size).")
        print("Pk_normal head:", Pk_normal[:5] if Pk_normal is not None else "None")


    print("Calculating power spectrum for Inverted Hierarchy...")
    kh_IH, Pk_inverted = get_power_spectrum('inverted')

    if kh_IH is None or Pk_inverted is None:
        print("Failed to compute power spectrum for Inverted Hierarchy. Exiting.")
        return
        
    if not (isinstance(Pk_inverted, np.ndarray) and len(Pk_inverted) == npoints_pk_param and np.all(Pk_inverted > 0)):
        print("Warning: Pk_inverted for Inverted Hierarchy seems invalid (e.g. non-positive values or wrong size).")
        print("Pk_inverted head:", Pk_inverted[:5] if Pk_inverted is not None else "None")


    # Verify k-values are the same (they should be)
    if not np.array_equal(kh_NH, kh_IH):
        print("Warning: k-values from Normal and Inverted hierarchy calculations differ. Using k_NH.")
        # This case should ideally not happen if parameters are consistent.
    
    k_values = kh_NH  # k in h/Mpc

    print("Calculating relative difference...")
    # Add a small epsilon to avoid division by zero, though P(k) should be > 0
    epsilon = 1e-30
    rel_diff = (Pk_inverted / (Pk_normal + epsilon)) - 1

    # Save results to CSV
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results_df = pd.DataFrame({
        'k': k_values,  # k is in h/Mpc
        'rel_diff': rel_diff
    })
    
    csv_filename = os.path.join(output_dir, "result.csv")
    results_df.to_csv(csv_filename, index=False)
    
    print("Results saved to " + csv_filename)
    print("First 5 rows of the result:")
    # Set pandas display options for detailed printing
    pd.set_option('display.float_format', '{:.6e}'.__mod__)
    print(results_df.head())
    
    print("\nSummary of computed values:")
    print("Number of k points: " + str(len(k_values)))
    print("k range (h/Mpc): " + ('%.2e' % k_values.min()) + " to " + ('%.2e' % k_values.max()))
    print("Relative difference range: " + ('%.3e' % rel_diff.min()) + " to " + ('%.3e' % rel_diff.max()))
    
    # Print some P(k) values for verification
    print("\nSample P(k) values for Normal Hierarchy (Mpc/h)^3:")
    for i in [0, len(Pk_normal)//4, len(Pk_normal)//2, 3*len(Pk_normal)//4, len(Pk_normal)-1]:
        print("k = " + ('%.4f' % k_values[i]) + " h/Mpc, P(k)_normal = " + ('%.4e' % Pk_normal[i]))

    print("\nSample P(k) values for Inverted Hierarchy (Mpc/h)^3:")
    for i in [0, len(Pk_inverted)//4, len(Pk_inverted)//2, 3*len(Pk_inverted)//4, len(Pk_inverted)-1]:
        print("k = " + ('%.4f' % k_values[i]) + " h/Mpc, P(k)_inverted = " + ('%.4e' % Pk_inverted[i]))


if __name__ == '__main__':
    try:
        calculate_matter_power_spectra()
    except ImportError:
        print("CAMB or other required packages not found. Please ensure they are installed.")
        # This is a fallback, the main error handling for CAMB itself is within the function.
    except Exception as e:
        print("An unexpected error occurred in the main script execution: " + str(e))
