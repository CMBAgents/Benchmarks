# filename: codebase/compute_pk_diff.py
import camb
from camb import model
import numpy as np
import pandas as pd
import os

def compute_pk_relative_difference(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Omega_b h^2 [dimensionless]
    omch2=0.122,            # Omega_c h^2 [dimensionless]
    mnu=0.11,               # Sum of neutrino masses [eV]
    As=2e-9,                # Scalar amplitude [dimensionless]
    ns=0.965,               # Scalar spectral index [dimensionless]
    omk=0.0,                # Curvature [dimensionless]
    z_pk=[0.0],             # Redshift(s) for P(k) [dimensionless]
    kmax_mpc=3.0,           # Maximum k for transfer [Mpc^-1]
    minkh=1e-4,             # Minimum k/h [h/Mpc]
    maxkh=2.0,              # Maximum k/h [h/Mpc]
    npoints=200,            # Number of k points [dimensionless]
    output_csv='data/result.csv' # Output CSV file path
):
    r"""
    Compute the relative difference in the linear matter power spectrum P(k) at z=0
    between normal and inverted neutrino hierarchies for a flat Lambda CDM cosmology.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter Omega_b h^2.
    omch2 : float
        Physical cold dark matter density parameter Omega_c h^2.
    mnu : float
        Sum of neutrino masses in eV.
    As : float
        Scalar amplitude of the primordial power spectrum.
    ns : float
        Scalar spectral index.
    omk : float
        Curvature parameter (0 for flat).
    z_pk : list of float
        Redshifts at which to compute the matter power spectrum.
    kmax_mpc : float
        Maximum wavenumber k (in Mpc^-1) for transfer function calculation.
    minkh : float
        Minimum wavenumber k/h (in h/Mpc) for output.
    maxkh : float
        Maximum wavenumber k/h (in h/Mpc) for output.
    npoints : int
        Number of k points (logarithmically spaced).
    output_csv : str
        Path to output CSV file.

    Returns
    -------
    None
        Saves the results to a CSV file and prints a summary to the console.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up CAMB parameters for normal hierarchy
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, neutrino_hierarchy='normal')
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=z_pk, kmax=kmax_mpc)
    pars.NonLinear = model.NonLinear_none

    # Compute P(k) for normal hierarchy
    results_normal = camb.get_results(pars)
    kh_vals, z_vals, pk_normal_allz = results_normal.get_matter_power_spectrum(
        minkh=minkh, maxkh=maxkh, npoints=npoints
    )
    pk_normal = pk_normal_allz[0]  # z=0

    # Set up CAMB parameters for inverted hierarchy (reuse pars)
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, neutrino_hierarchy='inverted')
    # Other settings remain unchanged

    # Compute P(k) for inverted hierarchy
    results_inverted = camb.get_results(pars)
    kh_vals_inv, z_vals_inv, pk_inverted_allz = results_inverted.get_matter_power_spectrum(
        minkh=minkh, maxkh=maxkh, npoints=npoints
    )
    pk_inverted = pk_inverted_allz[0]  # z=0

    # Check k consistency
    if not np.allclose(kh_vals, kh_vals_inv, rtol=1e-10, atol=1e-12):
        raise RuntimeError("k values for normal and inverted hierarchies do not match.")

    # Compute relative difference
    rel_diff = (pk_inverted / pk_normal) - 1

    # Save to CSV
    df = pd.DataFrame({'k': kh_vals, 'rel_diff': rel_diff})
    df.to_csv(output_csv, index=False)

    # Print summary
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print("Relative difference in linear matter power spectrum (P_inverted / P_normal - 1) at z=0")
    print("Cosmological parameters:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  Sum m_nu = " + str(mnu) + " eV")
    print("  As = " + str(As))
    print("  ns = " + str(ns))
    print("  Flat universe (Omega_k = 0)")
    print("k range: " + str(format(kh_vals[0], ".1e")) + " to " + str(format(kh_vals[-1], ".1e")) + " h/Mpc, " + str(npoints) + " points")
    print("Results saved to: " + output_csv)
    print("First 5 rows of results:")
    print(df.head(5).to_string(index=False))
    print("Last 5 rows of results:")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    compute_pk_relative_difference()
