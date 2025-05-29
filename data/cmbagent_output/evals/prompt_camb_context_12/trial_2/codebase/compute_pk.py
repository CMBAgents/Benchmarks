# filename: codebase/compute_pk.py
import camb
from camb import model
import numpy as np
import pandas as pd
import os

def compute_pk_relative_difference(
    H0=67.5,
    ombh2=0.022,
    omch2=0.122,
    mnu=0.11,
    As=2.0e-9,
    ns=0.965,
    omk=0.0,
    kh_min=1e-4,
    kh_max=2.0,
    n_points=200,
    kmax_calc=5.0,
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
        Physical baryon density parameter, Omega_b h^2 (dimensionless).
    omch2 : float
        Physical cold dark matter density parameter, Omega_c h^2 (dimensionless).
    mnu : float
        Sum of neutrino masses in eV.
    As : float
        Scalar amplitude of primordial power spectrum (dimensionless).
    ns : float
        Scalar spectral index (dimensionless).
    omk : float
        Curvature parameter (dimensionless, 0 for flat).
    kh_min : float
        Minimum wavenumber in h/Mpc.
    kh_max : float
        Maximum wavenumber in h/Mpc.
    n_points : int
        Number of k points (evenly spaced in k).
    kmax_calc : float
        Maximum k (in Mpc^-1) for CAMB internal calculation.
    output_dir : str
        Directory to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
        Saves a CSV file with columns:
            - k: Wavenumber in h/Mpc
            - rel_diff: Relative difference (P(k)_inverted / P(k)_normal - 1)
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare k grid (in h/Mpc)
    kh_grid = np.linspace(kh_min, kh_max, n_points)

    # Helper function to get P(k) for a given hierarchy
    def get_pk(hierarchy):
        pars = camb.set_params(
            H0=H0,
            ombh2=ombh2,
            omch2=omch2,
            mnu=mnu,
            omk=omk,
            As=As,
            ns=ns,
            neutrino_hierarchy=hierarchy,
            WantScalars=True,
            WantTensors=False,
            WantVectors=False,
            WantDerivedParameters=True,
            WantTransfer=True
        )
        pars.NonLinear = model.NonLinear_none
        pars.set_matter_power(redshifts=[0.0], kmax=kmax_calc)
        results = camb.get_results(pars)
        pk_interp = results.get_matter_power_interpolator(
            nonlinear=False,
            var1='delta_tot',
            var2='delta_tot',
            hubble_units=True,  # (Mpc/h)^3
            k_hunit=True,       # k in h/Mpc
            log_interp=True
        )
        pk = pk_interp.P(0.0, kh_grid)
        return pk

    # Compute P(k) for both hierarchies
    pk_normal = get_pk('normal')
    pk_inverted = get_pk('inverted')

    # Compute relative difference
    rel_diff = (pk_inverted / pk_normal) - 1.0

    # Save to CSV
    df = pd.DataFrame({'k': kh_grid, 'rel_diff': rel_diff})
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)

    # Print summary
    print("Relative difference in linear matter power spectrum (z=0) between inverted and normal neutrino hierarchies computed.")
    print("Cosmological parameters:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  Sum m_nu = " + str(mnu) + " eV")
    print("  As = " + str(As))
    print("  ns = " + str(ns))
    print("  Flat universe (Omega_k = 0)")
    print("k range: " + str(kh_min) + " < k < " + str(kh_max) + " h/Mpc, " + str(n_points) + " points")
    print("Results saved to: " + output_path)
    print("First 5 rows of the result:")
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    compute_pk_relative_difference()