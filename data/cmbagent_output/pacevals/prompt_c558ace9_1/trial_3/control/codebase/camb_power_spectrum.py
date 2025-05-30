# filename: codebase/camb_power_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

# Define cosmological parameters
H0_param = 67.5  # Hubble constant in km/s/Mpc
ombh2_param = 0.022  # Physical baryon density at z=0
omch2_param = 0.122  # Physical cold dark matter density at z=0
mnu_sum_param = 0.11  # Sum of neutrino masses in eV
As_param = 2.0e-9  # Scalar amplitude of primordial fluctuations
ns_param = 0.965  # Scalar spectral index
omk_param = 0.0  # Omega_K curvature parameter (0 for flat cosmology)
num_massive_neutrinos_param = 3 # Number of massive neutrino species

# Define k-range and redshift for power spectrum
kh_min_param = 1e-4  # Minimum k*h in h/Mpc
kh_max_param = 2.0  # Maximum k*h in h/Mpc
n_points_param = 200  # Number of k points
z_pk_param = 0.0  # Redshift for power spectrum calculation

# Output directory and filename
output_dir = "data"
csv_filename = os.path.join(output_dir, "result.csv")

def get_power_spectrum(hierarchy_model):
    """
    Computes the linear matter power spectrum for a given neutrino hierarchy.

    Parameters:
    hierarchy_model (str): Neutrino hierarchy model, 'normal' or 'inverted'.

    Returns:
    tuple: (kh, pk)
        kh (numpy.ndarray): Array of wavenumbers k/h (units: h/Mpc).
        pk (numpy.ndarray): Array of matter power spectrum P(k/h) (units: (Mpc/h)^3).
                           Returns (None, None) if CAMB calculation fails.
    """
    print("Calculating power spectrum for " + hierarchy_model + " hierarchy...")
    pars = camb.CAMBparams()

    # Set cosmological parameters
    # For 'normal' or 'inverted' hierarchy with num_massive_neutrinos > 1,
    # mnu is taken as the sum of masses.
    pars.set_cosmology(H0=H0_param, ombh2=ombh2_param, omch2=omch2_param,
                      mnu=mnu_sum_param, omk=omk_param,
                      neutrino_hierarchy=hierarchy_model,
                      num_massive_neutrinos=num_massive_neutrinos_param,
                      standard_neutrino_neff=3.046) # Standard N_eff for 3 neutrino species

    # Set initial power spectrum parameters
    pars.InitPower.set_params(As=As_param, ns=ns_param)

    # Configure matter power spectrum calculation
    # We want linear power spectrum at z=z_pk_param up to kh_max_param
    pars.set_matter_power(redshifts=[z_pk_param], kmax=kh_max_param, nonlinear=False)
    
    try:
        # Get results from CAMB
        results = camb.get_results(pars)

        # Extract linear matter power spectrum
        # kh_vals are k/h in h/Mpc
        # pk_output are P(k/h) in (Mpc/h)^3
        # For a single redshift, z_output is scalar and pk_output is a 1D array.
        kh_vals, z_output, pk_output = results.get_matter_power_spectrum(
            minkh=kh_min_param, maxkh=kh_max_param, npoints=n_points_param,
            var1='delta_tot', var2='delta_tot',  # Total matter fluctuations
            nonlinear=False,  # Request linear power spectrum
            hubble_units=True,  # P(k) in (Mpc/h)^3, k in h/Mpc if k_hunit=True
            k_hunit=True,  # k values are in h/Mpc
            Logarithmic=False # Ensure evenly spaced k values in linear scale
        )
        
        # Validate z_output and pk_output basic properties
        if not (isinstance(z_output, float) and np.isclose(z_output, z_pk_param)):
             print("Warning: Unexpected z_output from get_matter_power_spectrum: " + str(z_output))
        if not (isinstance(pk_output, np.ndarray) and pk_output.ndim == 1 and len(pk_output) == n_points_param):
             print("Warning: Unexpected type or shape for pk_output. Type: " + str(type(pk_output)) + ", Shape: " + str(getattr(pk_output, 'shape', 'N/A')))


        print("Power spectrum calculation for " + hierarchy_model + " hierarchy successful.")
        return kh_vals, pk_output

    except Exception as e:
        print("Error calculating power spectrum for " + hierarchy_model + " hierarchy:")
        # Print the full error
        import traceback
        print(traceback.format_exc())
        return None, None


# Main script execution
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print("Created directory: " + output_dir)
        except OSError as e:
            print("Error creating directory " + output_dir + ": " + str(e))
            # If directory creation fails, exit or handle appropriately
            exit()


    # Get P(k) for normal hierarchy
    kh_normal, pk_normal = get_power_spectrum('normal')

    # Get P(k) for inverted hierarchy
    kh_inverted, pk_inverted = get_power_spectrum('inverted')

    if kh_normal is not None and pk_normal is not None and \
       kh_inverted is not None and pk_inverted is not None:

        # Validation checks
        # 1. Check if k arrays are identical (they should be for the same request parameters)
        if not np.allclose(kh_normal, kh_inverted):
            print("Critical Error: k-arrays for normal and inverted hierarchies differ significantly.")
            print("Sample kh_normal: " + str(kh_normal[:5]))
            print("Sample kh_inverted: " + str(kh_inverted[:5]))
            print("Aborting due to k-vector mismatch.")
        else:
            print("\nk-vectors for normal and inverted hierarchies are consistent.")

            # 2. Check k-range and number of points from normal hierarchy (assuming they are identical)
            print("\nValidation of k-vector (from normal hierarchy calculation):")
            print("Number of k points: " + str(len(kh_normal)))
            print("Min k*h: " + str(kh_normal[0]) + " h/Mpc (requested: " + str(kh_min_param) + " h/Mpc)")
            print("Max k*h: " + str(kh_normal[-1]) + " h/Mpc (requested: " + str(kh_max_param) + " h/Mpc)")
            if len(kh_normal) != n_points_param:
                print("Warning: Number of k points (" + str(len(kh_normal)) + ") does not match requested (" + str(n_points_param) + ")")
            if not (np.isclose(kh_normal[0], kh_min_param)):
                 print("Warning: Min k*h (" + str(kh_normal[0]) + ") does not precisely match requested min (" + str(kh_min_param) + ")")
            if not (np.isclose(kh_normal[-1], kh_max_param)):
                 print("Warning: Max k*h (" + str(kh_normal[-1]) + ") does not precisely match requested max (" + str(kh_max_param) + ")")


            # 3. Check for NaNs or Infs in power spectra
            if np.any(np.isnan(pk_normal)) or np.any(np.isinf(pk_normal)):
                print("Warning: NaNs or Infs found in P(k) for normal hierarchy.")
            if np.any(np.isnan(pk_inverted)) or np.any(np.isinf(pk_inverted)):
                print("Warning: NaNs or Infs found in P(k) for inverted hierarchy.")

            # Calculate relative difference: (P(k)_inverted / P(k)_normal - 1)
            # Avoid division by zero warning and handle cases where pk_normal is zero or negative.
            rel_diff = np.empty_like(pk_normal)
            rel_diff.fill(np.nan) # Default to NaN for problematic points

            # P(k) should be positive. Use a small epsilon for safe division.
            epsilon = 1e-30 
            positive_pk_normal_mask = pk_normal > epsilon

            num_non_positive_pk_normal = np.sum(~positive_pk_normal_mask)
            if num_non_positive_pk_normal > 0:
                print("Warning: " + str(num_non_positive_pk_normal) +
                      " P(k) values in normal hierarchy are <= " + str(epsilon) + ".")
                print("Relative difference will be NaN at these k points.")
            
            rel_diff[positive_pk_normal_mask] = \
                (pk_inverted[positive_pk_normal_mask] / pk_normal[positive_pk_normal_mask]) - 1

            # Create DataFrame
            results_df = pd.DataFrame({
                'k': kh_normal,  # Wavenumber in h/Mpc
                'rel_diff': rel_diff  # Relative difference (dimensionless)
            })

            # Save to CSV
            try:
                results_df.to_csv(csv_filename, index=False, float_format='%.8e')
                print("\nResults saved to: " + csv_filename)
            except Exception as e:
                print("Error saving results to CSV (" + csv_filename + "): " + str(e))

            # Print summary of results
            print("\nSummary of calculated power spectra and relative difference:")
            if pk_normal is not None:
                print("Min P(k) Normal: " + str(np.nanmin(pk_normal)) + " (Mpc/h)^3")
                print("Max P(k) Normal: " + str(np.nanmax(pk_normal)) + " (Mpc/h)^3")
            if pk_inverted is not None:
                print("Min P(k) Inverted: " + str(np.nanmin(pk_inverted)) + " (Mpc/h)^3")
                print("Max P(k) Inverted: " + str(np.nanmax(pk_inverted)) + " (Mpc/h)^3")
            
            if np.any(np.isfinite(rel_diff)):
                print("Min Relative Difference: " + str(np.nanmin(rel_diff)))
                print("Max Relative Difference: " + str(np.nanmax(rel_diff)))
            else:
                print("Relative Difference array contains only NaNs or Infs, or is empty.")
                
            print("\nFirst few rows of the output data (result.csv):")
            # Ensure DataFrame is not empty before printing head
            if not results_df.empty:
                try:
                    # Using to_string() for better console formatting of a few rows
                    print(results_df.head().to_string())
                except Exception as e:
                    print("Error printing DataFrame head: " + str(e))
            else:
                print("Result DataFrame is empty.")
    else:
        print("\nOne or both power spectrum calculations failed. Cannot proceed to calculate relative difference or save results.")

    print("\nScript finished.")
