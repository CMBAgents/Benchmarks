# filename: codebase/camb_pk_difference.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model

def calculate_and_save_pk_diff():
    r"""
    Calculates the relative difference in the linear matter power spectrum P(k)
    at redshift z=0 between normal and inverted neutrino hierarchy models.

    The calculation uses CAMB with specified flat Lambda CDM cosmological parameters:
    - Hubble constant (H0): 67.5 km/s/Mpc
    - Baryon density (ombh2): 0.022
    - Cold dark matter density (omch2): 0.122
    - Neutrino mass sum (sum_mnu): 0.11 eV
    - Scalar amplitude (As): 2e-9
    - Scalar spectral index (ns): 0.965

    The linear matter power spectrum P(k/h) in units of (Mpc/h)^3 is computed
    for 200 linearly spaced k/h values in the range 1e-4 < k/h < 2.0 (h/Mpc).

    The relative difference is calculated as (P(k)_inverted / P(k)_normal - 1).
    Results are saved to 'data/result.csv' with columns 'k' and 'rel_diff'.
    """

    # Cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Physical baryon density
    omch2 = 0.122  # Physical cold dark matter density
    sum_mnu = 0.11  # Sum of neutrino masses in eV
    As = 2.0e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    omk = 0.0  # Curvature parameter (0 for flat)
    tau = None  # Optical depth to reionization (CAMB will calculate if None)

    # k-range for power spectrum (k/h values)
    minkh = 1e-4  # Minimum k/h in h/Mpc
    maxkh = 2.0  # Maximum k/h in h/Mpc
    npoints = 200  # Number of k points
    
    # Redshift
    redshifts = [0.0]

    # --- Normal Hierarchy ---
    pars_normal = camb.CAMBparams()
    pars_normal.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=sum_mnu, omk=omk, tau=tau)
    pars_normal.InitPower.set_params(As=As, ns=ns)
    pars_normal.set_matter_power(redshifts=redshifts, kmax=maxkh + 0.5)  # kmax slightly larger for interpolation
    pars_normal.NonLinear = model.NonLinear_none  # Linear power spectrum
    pars_normal.WantCls = False  # We don't need CMB Cls
    
    pars_normal.num_massive_neutrinos = 3
    pars_normal.nu_mass_model = model.nu_mass_normal  # Normal hierarchy

    results_normal = camb.get_results(pars_normal)
    kh_normal, z_normal, pk_normal_array = results_normal.get_matter_power_spectrum(
        minkh=minkh, maxkh=maxkh, npoints=npoints, log_spacing=False
    )
    pk_normal = pk_normal_array[0]  # P(k) at z=0

    # --- Inverted Hierarchy ---
    pars_inverted = camb.CAMBparams()
    pars_inverted.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=sum_mnu, omk=omk, tau=tau)
    pars_inverted.InitPower.set_params(As=As, ns=ns)
    pars_inverted.set_matter_power(redshifts=redshifts, kmax=maxkh + 0.5)
    pars_inverted.NonLinear = model.NonLinear_none
    pars_inverted.WantCls = False
    
    pars_inverted.num_massive_neutrinos = 3
    pars_inverted.nu_mass_model = model.nu_mass_inverted  # Inverted hierarchy

    results_inverted = camb.get_results(pars_inverted)
    kh_inverted, z_inverted, pk_inverted_array = results_inverted.get_matter_power_spectrum(
        minkh=minkh, maxkh=maxkh, npoints=npoints, log_spacing=False
    )
    pk_inverted = pk_inverted_array[0]  # P(k) at z=0

    # Ensure k-values are the same (they should be by construction)
    if not np.allclose(kh_normal, kh_inverted):
        print("Warning: k-vectors for normal and inverted hierarchies differ significantly.")
        # Fallback to using kh_normal if they are different, though this shouldn't happen.
    
    k_values = kh_normal  # These are k/h in h/Mpc

    # Calculate relative difference
    # (P(k)_inverted / P(k)_normal - 1)
    # Handle potential division by zero, though P(k) should be positive
    rel_diff = np.zeros_like(pk_normal)
    mask_pk_normal_nonzero = pk_normal != 0
    
    # Calculate relative difference only where pk_normal is not zero
    rel_diff[mask_pk_normal_nonzero] = (pk_inverted[mask_pk_normal_nonzero] / \
                                        pk_normal[mask_pk_normal_nonzero]) - 1
    
    # For cases where pk_normal is zero (should not happen for P(k) > 0)
    # If pk_inverted is also zero, rel_diff is 0 (already set by np.zeros_like)
    # If pk_inverted is non-zero and pk_normal is zero, this implies infinite difference.
    # We can set it to NaN or a large number, or handle as per specific requirements.
    # For now, it remains 0 from initialization if pk_normal is 0.
    # A more robust way for physical spectra where P(k) > 0 is direct division.
    if np.any(pk_normal <= 0):
        print("Warning: Some P(k) values for normal hierarchy are zero or negative.")
    
    rel_diff = (pk_inverted / pk_normal) - 1


    # Create DataFrame
    df_results = pd.DataFrame({
        'k': k_values,  # k in h/Mpc
        'rel_diff': rel_diff
    })

    # Save to CSV
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    csv_filepath = os.path.join(output_dir, "result.csv")
    df_results.to_csv(csv_filepath, index=False)

    print("Successfully calculated matter power spectra and relative differences.")
    print("Results saved to: " + csv_filepath)
    print("\n--- Data Summary ---")
    print("Number of k points: " + str(len(df_results)))
    print("k range (h/Mpc): " + str(df_results['k'].min()) + " to " + str(df_results['k'].max()))
    
    print("\nFirst 5 rows of the data:")
    # Pandas to_string() can be long, let's print manually for concise output
    for i in range(min(5, len(df_results))):
        print("k=" + str(df_results['k'].iloc[i]) + ", rel_diff=" + str(df_results['rel_diff'].iloc[i]))

    print("\nLast 5 rows of the data:")
    for i in range(max(0, len(df_results)-5), len(df_results)):
         print("k=" + str(df_results['k'].iloc[i]) + ", rel_diff=" + str(df_results['rel_diff'].iloc[i]))

    print("\nSummary statistics for relative difference (rel_diff):")
    # Manually print describe() like output to avoid f-strings and .format
    desc = df_results['rel_diff'].describe()
    print("Count: " + str(desc['count']))
    print("Mean: " + str(desc['mean']))
    print("Std: " + str(desc['std']))
    print("Min: " + str(desc['min']))
    print("25%: " + str(desc['25%']))
    print("50%: " + str(desc['50%']))
    print("75%: " + str(desc['75%']))
    print("Max: " + str(desc['max']))


if __name__ == "__main__":
    try:
        calculate_and_save_pk_diff()
    except ImportError as e:
        print("ImportError: " + str(e))
        print("Please ensure CAMB is installed and accessible.")
    except Exception as e:
        print("An error occurred during the calculation:")
        import traceback
        traceback.print_exc()
