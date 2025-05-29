# filename: codebase/power_spectrum_comparison.py
import camb
from camb import model
import numpy as np
import pandas as pd
import os

def compute_pk(params, kh_array, z=0.0):
    r"""
    Compute the linear matter power spectrum P(k) at redshift z for given CAMB parameters.

    Parameters
    ----------
    params : camb.CAMBparams
        CAMB parameters object with cosmology and power spectrum settings.
    kh_array : np.ndarray
        Array of wavenumbers in units of h/Mpc.
    z : float
        Redshift at which to compute the power spectrum.

    Returns
    -------
    pk : np.ndarray
        Linear matter power spectrum values at the specified k and z, in (Mpc/h)^3.
    """
    results = camb.get_results(params)
    pk_interp = results.get_matter_power_interpolator(
        nonlinear=False,
        hubble_units=True,
        k_hunit=True,
        var1='delta_tot',
        var2='delta_tot'
    )
    pk = pk_interp.P(z, kh_array)
    return pk

def main():
    r"""
    Main function to compute and save the relative difference in the linear matter power spectrum
    between inverted and normal neutrino hierarchies for specified cosmological parameters.
    Results are saved to 'data/result.csv' with columns:
        - k: Wavenumber in h/Mpc (float, 200 values)
        - rel_diff: Relative difference (P_inverted / P_normal - 1) (float)
    """
    # Output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_csv = os.path.join(output_dir, "result.csv")

    # Cosmological parameters (units in comments)
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # CDM density [dimensionless]
    mnu = 0.11  # Sum of neutrino masses [eV]
    As = 2.0e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]
    omk = 0.0  # Curvature (flat universe)

    # k range (h/Mpc)
    kh_min = 1e-4
    kh_max = 2.0
    n_k = 200
    kh_array = np.linspace(kh_min, kh_max, n_k)

    # Redshift
    z = 0.0

    # kmax for CAMB (in Mpc^-1)
    kmax_camb = 3.0

    # --- Normal hierarchy ---
    pars_normal = camb.CAMBparams()
    pars_normal.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, neutrino_hierarchy='normal')
    pars_normal.InitPower.set_params(As=As, ns=ns)
    pars_normal.set_matter_power(redshifts=[z], kmax=kmax_camb)
    pars_normal.NonLinear = model.NonLinear_none

    P_normal = compute_pk(pars_normal, kh_array, z=z)

    # --- Inverted hierarchy ---
    pars_inverted = camb.CAMBparams()
    pars_inverted.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, neutrino_hierarchy='inverted')
    pars_inverted.InitPower.set_params(As=As, ns=ns)
    pars_inverted.set_matter_power(redshifts=[z], kmax=kmax_camb)
    pars_inverted.NonLinear = model.NonLinear_none

    P_inverted = compute_pk(pars_inverted, kh_array, z=z)

    # --- Relative difference ---
    rel_diff = (P_inverted / P_normal) - 1

    # --- Save to CSV ---
    df = pd.DataFrame({'k': kh_array, 'rel_diff': rel_diff})
    df.to_csv(output_csv, index=False)

    # Print summary
    print("Relative difference in linear matter power spectrum (inverted vs normal hierarchy) computed at z=0.")
    print("k (h/Mpc) range: " + ("%.4e" % kh_min) + " to " + ("%.4e" % kh_max) + ", number of points: " + str(n_k))
    print("Results saved to " + output_csv)
    print("First 5 rows of the result:")
    print(df.head())


if __name__ == "__main__":
    main()
