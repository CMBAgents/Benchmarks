# filename: codebase/camb_neutrino_pk_comparison.py
r"""
Compute the relative difference in the linear matter power spectrum P(k) at z=0
between normal and inverted neutrino hierarchies for a flat Lambda CDM cosmology
using CAMB, and save the results in a CSV file.

All units are annotated in the code.

Output file: data/result.csv
Columns:
    k        : Wavenumber in h/Mpc
    rel_diff : Relative difference (P_inverted / P_normal - 1)
"""

import camb
from camb import model
import numpy as np
import pandas as pd
import os

def compute_pk(
    H0,
    ombh2,
    omch2,
    mnu_sum,
    As_val,
    ns_val,
    tau_val,
    neutrino_hierarchy,
    kh_vals,
    kmax_for_camb
):
    r"""
    Compute the linear matter power spectrum P(k) at z=0 for given cosmological parameters and neutrino hierarchy.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density, Omega_b h^2.
    omch2 : float
        Physical cold dark matter density, Omega_c h^2.
    mnu_sum : float
        Sum of neutrino masses in eV.
    As_val : float
        Scalar amplitude of primordial power spectrum.
    ns_val : float
        Scalar spectral index.
    tau_val : float
        Optical depth to reionization.
    neutrino_hierarchy : str
        'normal' or 'inverted'.
    kh_vals : np.ndarray
        Array of wavenumbers in h/Mpc at which to evaluate P(k).
    kmax_for_camb : float
        Maximum k (in Mpc^-1) for CAMB internal calculation.

    Returns
    -------
    pk : np.ndarray
        Linear matter power spectrum at z=0, in (Mpc/h)^3, evaluated at kh_vals.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu_sum,
        omk=0.0,
        neutrino_hierarchy=neutrino_hierarchy,
        tau=tau_val
    )
    pars.InitPower.set_params(As=As_val, ns=ns_val)
    pars.set_matter_power(redshifts=[0.0], kmax=kmax_for_camb)
    pars.NonLinear = model.NonLinear_none

    results = camb.get_results(pars)
    PK = results.get_matter_power_interpolator(
        nonlinear=False,
        var1='delta_tot',
        var2='delta_tot',
        hubble_units=True,  # Output P(k) in (Mpc/h)^3
        k_hunit=True        # Input k in h/Mpc
    )
    pk = PK.P(0.0, kh_vals)
    return pk

def main():
    r"""
    Main routine to compute and save the relative difference in the linear matter power spectrum
    between inverted and normal neutrino hierarchies.
    """
    # Cosmological parameters (all units in code comments)
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Omega_b h^2 [dimensionless]
    omch2 = 0.122  # Omega_c h^2 [dimensionless]
    mnu_sum = 0.11  # Sum of neutrino masses [eV]
    As_val = 2e-9  # Scalar amplitude [dimensionless]
    ns_val = 0.965  # Scalar spectral index [dimensionless]
    tau_val = 0.06  # Optical depth [dimensionless]

    h = H0 / 100.0  # Dimensionless Hubble parameter

    # Wavenumber array: 200 evenly spaced values in 1e-4 < k h < 2 (h/Mpc)
    kh_vals = np.linspace(1e-4, 2.0, 200)  # [h/Mpc]

    # kmax for CAMB internal calculation (in Mpc^-1)
    kmax_for_camb = (kh_vals.max() / h) * 1.05  # [Mpc^-1], slightly above max(kh)/h

    # Compute P(k) for normal hierarchy
    pk_normal = compute_pk(
        H0, ombh2, omch2, mnu_sum, As_val, ns_val, tau_val,
        'normal', kh_vals, kmax_for_camb
    )

    # Compute P(k) for inverted hierarchy
    pk_inverted = compute_pk(
        H0, ombh2, omch2, mnu_sum, As_val, ns_val, tau_val,
        'inverted', kh_vals, kmax_for_camb
    )

    # Compute relative difference
    rel_diff = (pk_inverted / pk_normal) - 1

    # Prepare output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save results to CSV
    output_path = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({
        "k": kh_vals,         # [h/Mpc]
        "rel_diff": rel_diff  # [dimensionless]
    })
    df.to_csv(output_path, index=False)

    # Print summary to console
    print("Relative difference in linear matter power spectrum (P_inverted / P_normal - 1) at z=0")
    print("Cosmological parameters:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  Sum m_nu = " + str(mnu_sum) + " eV")
    print("  A_s = " + str(As_val))
    print("  n_s = " + str(ns_val))
    print("  tau = " + str(tau_val))
    print("k range: " + str(kh_vals[0]) + " to " + str(kh_vals[-1]) + " h/Mpc, 200 points")
    print("Results saved to " + output_path)
    print("First 5 rows of the result:")
    print(df.head())


if __name__ == "__main__":
    main()
