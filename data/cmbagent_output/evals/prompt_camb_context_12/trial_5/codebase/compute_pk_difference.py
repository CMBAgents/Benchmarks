# filename: codebase/compute_pk_difference.py
import camb
from camb import model
import numpy as np
import pandas as pd
import os

def compute_pk_difference(
    H0=67.5,
    ombh2=0.022,
    omch2=0.122,
    mnu=0.11,
    As=2e-9,
    ns=0.965,
    omk=0.0,
    z_pk=[0.0],
    kmax_mpc=3.0,
    minkh=1e-4,
    maxkh=2.0,
    npoints=200,
    output_dir="data/",
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
        Physical baryon density, Omega_b h^2.
    omch2 : float
        Physical cold dark matter density, Omega_c h^2.
    mnu : float
        Sum of neutrino masses in eV.
    As : float
        Scalar amplitude of the primordial power spectrum.
    ns : float
        Scalar spectral index.
    omk : float
        Curvature density parameter (0 for flat).
    z_pk : list of float
        Redshifts at which to compute the power spectrum.
    kmax_mpc : float
        Maximum k (in Mpc^-1) for CAMB internal calculation.
    minkh : float
        Minimum k/h (in h/Mpc) for output.
    maxkh : float
        Maximum k/h (in h/Mpc) for output.
    npoints : int
        Number of k points (logarithmically spaced).
    output_dir : str
        Directory to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
        Saves a CSV file with columns 'k' (h/Mpc) and 'rel_diff' (dimensionless).
    """
    if not os.path.exists(output_dir):
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

    results_inverted = camb.get_results(pars)
    kh_vals_inv, z_vals_inv, pk_inverted_allz = results_inverted.get_matter_power_spectrum(
        minkh=minkh, maxkh=maxkh, npoints=npoints
    )
    pk_inverted = pk_inverted_allz[0]  # z=0

    # Check k consistency
    if not np.allclose(kh_vals, kh_vals_inv, rtol=1e-10, atol=1e-12):
        raise ValueError("k values for normal and inverted hierarchies do not match.")

    # Compute relative difference
    rel_diff = (pk_inverted / pk_normal) - 1

    # Save to CSV
    df = pd.DataFrame({
        "k": kh_vals,         # h/Mpc
        "rel_diff": rel_diff  # dimensionless
    })
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)

    # Print summary
    print("Relative difference in linear matter power spectrum (inverted/normal - 1) at z=0 computed.")
    print("k (h/Mpc) range: " + format(kh_vals[0], ".2e") + " to " + format(kh_vals[-1], ".2e"))
    print("Number of k points: " + str(len(kh_vals)))
    print("Results saved to: " + output_path)
    print("First 5 rows of the result:")
    print(df.head())


if __name__ == "__main__":
    compute_pk_difference()
