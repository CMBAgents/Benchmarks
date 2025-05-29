# filename: codebase/compute_pk_relative_difference.py
import camb
from camb import model
import numpy as np
import pandas as pd
import os

def compute_pk_relative_difference(
    H0=67.5,
    ombh2=0.022,
    omch2=0.122,
    mnu_sum=0.11,
    As=2.0e-9,
    ns=0.965,
    omk=0.0,
    kh_min=1e-4,
    kh_max=2.0,
    n_k=200,
    z=0.0,
    kmax_camb=3.0,
    output_dir="data",
    output_filename="result.csv"
):
    r"""
    Compute the relative difference in the linear matter power spectrum P(k) at z=0
    between normal and inverted neutrino hierarchies for a flat Lambda CDM cosmology.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter \Omega_b h^2 (dimensionless).
    omch2 : float
        Physical cold dark matter density parameter \Omega_c h^2 (dimensionless).
    mnu_sum : float
        Sum of neutrino masses in eV.
    As : float
        Scalar amplitude of the primordial power spectrum (dimensionless).
    ns : float
        Scalar spectral index (dimensionless).
    omk : float
        Curvature density parameter (dimensionless, 0 for flat).
    kh_min : float
        Minimum wavenumber in h/Mpc.
    kh_max : float
        Maximum wavenumber in h/Mpc.
    n_k : int
        Number of k points.
    z : float
        Redshift at which to compute the power spectrum.
    kmax_camb : float
        Maximum k (in Mpc^-1) for CAMB internal calculation.
    output_dir : str
        Directory to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
        Saves the results to a CSV file and prints a summary.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    kh_array = np.linspace(kh_min, kh_max, n_k)  # [h/Mpc]

    # --- Normal hierarchy ---
    pars_normal = camb.CAMBparams()
    pars_normal.set_cosmology(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu_sum,
        omk=omk,
        neutrino_hierarchy='normal'
    )
    pars_normal.InitPower.set_params(As=As, ns=ns)
    pars_normal.set_matter_power(redshifts=[z], kmax=kmax_camb)
    pars_normal.NonLinear = model.NonLinear_none
    results_normal = camb.get_results(pars_normal)
    PK_normal = results_normal.get_matter_power_interpolator(
        nonlinear=False,
        hubble_units=True,
        k_hunit=True,
        var1='delta_tot',
        var2='delta_tot'
    )
    P_normal = PK_normal.P(z, kh_array)  # [(Mpc/h)^3]

    # --- Inverted hierarchy ---
    pars_invert = camb.CAMBparams()
    pars_invert.set_cosmology(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu_sum,
        omk=omk,
        neutrino_hierarchy='inverted'
    )
    pars_invert.InitPower.set_params(As=As, ns=ns)
    pars_invert.set_matter_power(redshifts=[z], kmax=kmax_camb)
    pars_invert.NonLinear = model.NonLinear_none
    results_invert = camb.get_results(pars_invert)
    PK_invert = results_invert.get_matter_power_interpolator(
        nonlinear=False,
        hubble_units=True,
        k_hunit=True,
        var1='delta_tot',
        var2='delta_tot'
    )
    P_invert = PK_invert.P(z, kh_array)  # [(Mpc/h)^3]

    # --- Relative difference ---
    rel_diff = (P_invert / P_normal) - 1

    # --- Save to CSV ---
    output_path = os.path.join(output_dir, output_filename)
    df = pd.DataFrame({'k': kh_array, 'rel_diff': rel_diff})
    df.to_csv(output_path, index=False)

    # Print summary
    print("Relative difference in linear matter power spectrum (inverted vs normal hierarchy) computed.")
    print("k range: " + str(format(kh_min, '.2e')) + " to " + str(format(kh_max, '.2e')) + " h/Mpc, " + str(n_k) + " points")
    print("Results saved to " + output_path)
    print("First 5 rows:")
    print(df.head())


if __name__ == "__main__":
    compute_pk_relative_difference()