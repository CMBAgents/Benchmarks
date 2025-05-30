# filename: codebase/calculate_matter_power_spectra.py
import camb
import numpy as np
import pandas as pd
import os
import time


def calculate_matter_power_spectra():
    """
    Calculates the linear matter power spectrum for normal and inverted
    neutrino hierarchies using CAMB, computes the relative difference,
    and saves the results to a CSV file.

    The cosmological parameters are:
    H0 = 67.5 km/s/Mpc
    ombh2 = 0.022
    omch2 = 0.122
    mnu = 0.11 eV
    As = 2e-9
    ns = 0.965
    omk = 0 (flat universe)

    The output k-values for the CSV are k_phys * h, ranging from 1e-4 to 2.0 (h/Mpc).
    The power spectrum P(k) is P(k_phys) in units of (Mpc/h)^3.
    The relative difference is (P_inverted(k_phys) / P_normal(k_phys) - 1).
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Physical baryon density
    omch2 = 0.122  # Physical cold dark matter density
    sum_mnu = 0.11  # Sum of neutrino masses in eV
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    omk = 0.0  # Curvature density parameter (0 for flat)

    h = H0 / 100.0  # Dimensionless Hubble parameter

    # k-range for the output CSV file (k_csv = k_phys * h)
    # k_csv has units of h/Mpc, which is equivalent to Mpc^-1 if h is used to scale k_phys.
    # The problem states "k h < 2 (Mpc^-1)", so k_csv_max = 2.0.
    k_min_csv = 1e-4  # min k_phys * h
    k_max_csv = 2.0   # max k_phys * h
    npoints = 200   # Number of k points

    # Derived parameters for CAMB's get_matter_power_spectrum
    # We want k_csv = k_camb * h, where k_camb = k_phys.
    # So k_camb (k_phys) should be in [k_min_csv/h, k_max_csv/h].
    # k_camb = k'_grid * h, where k'_grid = k_phys/h (input to CAMB's interpolation).
    # So k'_grid should be in [k_min_csv/h^2, k_max_csv/h^2].
    # These k'_grid values are the minkh, maxkh parameters for get_matter_power_spectrum
    # when k_hunit=True.
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
    print("kmax for set_matter_power (k_phys): " + str(kmax_internal_camb))
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
            # redshifts=[0.] for z=0
            # kmax is physical k_max in Mpc^-1
            pars.set_matter_power(redshifts=[0.], kmax=kmax_internal_camb)
            pars.NonLinear = camb.model.NonLinear_none # Ensure linear power spectrum

            results = camb.get_results(pars)
            
            # Get matter power spectrum P(k)
            # hubble_units=True: Pk in (Mpc/h)^3, k_camb in Mpc^-1 (k_phys)
            # k_hunit=True: minkh, maxkh are k_phys/h
            # So, k_camb (returned k) = (k_phys/h)_grid * h = k_phys
            # And pk_camb (returned Pk) = P(k_phys)
            k_camb, _, pk_camb = results.get_matter_power_spectrum(
                minkh=minkh_param_camb, maxkh=maxkh_param_camb, npoints=npoints,
                var1='delta_tot', var2='delta_tot', # Total matter power spectrum
                hubble_units=True, k_hunit=True, nonlinear=False
            )
            # k_values_for_csv should be k_phys * h
            k_values_for_csv = k_camb * h
            return k_values_for_csv, pk_camb[0] # pk_camb is 2D (z,k), we take z=0
        except Exception as e:
            print("Error calculating power spectrum for " + hierarchy_model + " hierarchy: " + str(e))
            raise

    try:
        print("\nCalculating for Normal Hierarchy...")
        k_normal_csv, pk_normal = get_pk_for_hierarchy('normal')
        
        print("Calculating for Inverted Hierarchy...")
        k_inverted_csv, pk_inverted = get_pk_for_hierarchy('inverted')

        # Check if k-arrays are reasonably close (they should be identical)
        if not np.allclose(k_normal_csv, k_inverted_csv):
            print("Warning: k-arrays from normal and inverted hierarchy calculations differ significantly.")
            # Use k_normal_csv as the reference if they differ slightly due to precision
        
        k_values_output = k_normal_csv # k_phys * h in h/Mpc

        # Calculate relative difference: (P_inverted / P_normal - 1)
        # pk_normal and pk_inverted are P(k_phys)
        # The problem asks for P(k_csv/h) = P(k_phys)
        with np.errstate(divide='ignore', invalid='ignore'): # Handle potential division by zero or NaN
            rel_diff = (pk_inverted / pk_normal) - 1
            # Replace NaNs or Infs that might occur if pk_normal is zero
            rel_diff[~np.isfinite(rel_diff)] = 0 


        # Create DataFrame
        df_results = pd.DataFrame({
            'k_h_Mpc': k_values_output, # k in h/Mpc
            'rel_diff_Pinverted_Pnormal_minus_1': rel_diff
        })

        # Create data directory if it doesn't exist
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print("Created directory: " + data_dir)

        # Save to CSV
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(data_dir, "result_neutrino_pk_diff_" + timestamp + ".csv")
        df_results.to_csv(filename, index=False)
        print("\nResults saved to: " + filename)

        print("\nFirst 5 rows of the results:")
        print(df_results.head().to_string())
        print("\nLast 5 rows of the results:")
        print(df_results.tail().to_string())
        
        # Print some statistics of the relative difference
        print("\nStatistics for relative difference:")
        print("Min rel_diff: " + str(np.min(rel_diff)))
        print("Max rel_diff: " + str(np.max(rel_diff)))
        print("Mean rel_diff: " + str(np.mean(rel_diff)))
        
        # Verify k-range
        print("\nOutput k-range (h/Mpc):")
        print("Min k: " + str(np.min(k_values_output)))
        print("Max k: " + str(np.max(k_values_output)))
        print("Number of k points: " + str(len(k_values_output)))


    except Exception as e:
        print("An error occurred during the calculation: " + str(e))


if __name__ == '__main__':
    calculate_matter_power_spectra()
