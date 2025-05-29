# filename: codebase/compute_pk_relative_difference.py
import camb
from camb import model
import numpy as np
import pandas as pd
import os


def compute_pk_relative_difference():
    r"""
    Compute the relative difference in the linear matter power spectrum P(k) at z=0
    between normal and inverted neutrino hierarchies for a flat Lambda CDM cosmology
    using CAMB, and save the results to a CSV file.

    The cosmological parameters used are:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Neutrino mass sum (mnu): 0.11 eV
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965

    The power spectrum is computed for 200 evenly spaced k values in the range
    1e-4 < k h < 2 (in h/Mpc), and the relative difference is calculated as
    (P_inverted / P_normal - 1).

    The results are saved to 'data/result.csv' with columns:
        - k: Wavenumber in h/Mpc
        - rel_diff: Relative difference in the matter power spectrum
    """
    # Output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cosmological parameters (units in comments)
    H0 = 67.5                # Hubble constant [km/s/Mpc]
    ombh2 = 0.022            # Baryon density [dimensionless]
    omch2 = 0.122            # Cold dark matter density [dimensionless]
    mnu = 0.11               # Neutrino mass sum [eV]
    As = 2.0e-9              # Scalar amplitude [dimensionless]
    ns = 0.965               # Scalar spectral index [dimensionless]
    omk = 0.0                # Curvature [dimensionless]

    # k range in h/Mpc
    kmin = 1e-4              # Minimum wavenumber [h/Mpc]
    kmax = 2.0               # Maximum wavenumber [h/Mpc]
    nk = 200                 # Number of k points
    kh = np.linspace(kmin, kmax, nk)  # Wavenumber array [h/Mpc]

    # Redshift for P(k)
    z = 0.0                  # Redshift [dimensionless]

    # kmax for CAMB (in Mpc^-1, must be >= max(kh) * h)
    kmax_camb = 3.0          # Maximum k for CAMB [Mpc^-1]

    # --- Normal hierarchy ---
    pars_n = camb.CAMBparams()
    pars_n.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, neutrino_hierarchy='normal')
    pars_n.InitPower.set_params(As=As, ns=ns)
    pars_n.set_matter_power(redshifts=[z], kmax=kmax_camb)
    pars_n.NonLinear = model.NonLinear_none
    results_n = camb.get_results(pars_n)
    PK_n = results_n.get_matter_power_interpolator(
        nonlinear=False,
        hubble_units=True,
        k_hunit=True,
        var1='delta_tot',
        var2='delta_tot'
    )
    P_n = PK_n.P(z, kh)  # [Mpc/h]^3

    # --- Inverted hierarchy ---
    pars_i = camb.CAMBparams()
    pars_i.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, neutrino_hierarchy='inverted')
    pars_i.InitPower.set_params(As=As, ns=ns)
    pars_i.set_matter_power(redshifts=[z], kmax=kmax_camb)
    pars_i.NonLinear = model.NonLinear_none
    results_i = camb.get_results(pars_i)
    PK_i = results_i.get_matter_power_interpolator(
        nonlinear=False,
        hubble_units=True,
        k_hunit=True,
        var1='delta_tot',
        var2='delta_tot'
    )
    P_i = PK_i.P(z, kh)  # [Mpc/h]^3

    # --- Relative difference ---
    rel_diff = (P_i / P_n) - 1

    # --- Save to CSV ---
    output_path = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'k': kh, 'rel_diff': rel_diff})
    df.to_csv(output_path, index=False)

    # Print summary
    print("Relative difference in linear matter power spectrum (P_inverted / P_normal - 1) computed for 200 k values.")
    print("k range: " + str(kmin) + " to " + str(kmax) + " h/Mpc")
    print("Results saved to " + output_path)
    print("First 5 rows of the result:")
    print(df.head())


if __name__ == "__main__":
    compute_pk_relative_difference()