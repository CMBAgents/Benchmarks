# filename: codebase/compute_pk_diff.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model

def ensure_data_dir():
    r"""Ensure the 'data' directory exists for output files."""
    if not os.path.exists("data"):
        os.makedirs("data")

def compute_pk(
    H0, ombh2, omch2, mnu, As, ns, omk, hierarchy, kmax_calc, kh_grid
):
    r"""
    Compute the linear matter power spectrum P(k) at z=0 for a given neutrino hierarchy.

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
        Scalar amplitude.
    ns : float
        Scalar spectral index.
    omk : float
        Curvature parameter (0 for flat).
    hierarchy : str
        'normal' or 'inverted' neutrino mass hierarchy.
    kmax_calc : float
        Maximum k (in Mpc^-1) for CAMB internal calculation.
    kh_grid : ndarray
        1D array of k values in h/Mpc at which to evaluate P(k).

    Returns
    -------
    pk : ndarray
        1D array of P(k) in (Mpc/h)^3 at z=0, evaluated at kh_grid.
    """
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
    PK = results.get_matter_power_interpolator(
        nonlinear=False,
        var1='delta_tot',
        var2='delta_tot',
        hubble_units=True,
        k_hunit=True,
        log_interp=True
    )
    pk = PK.P(0.0, kh_grid)
    return pk

def main():
    r"""Main routine to compute and save the relative difference in P(k) between hierarchies."""
    ensure_data_dir()

    # Cosmological parameters (units in comments)
    H0 = 67.5  # km/s/Mpc
    ombh2 = 0.022  # dimensionless
    omch2 = 0.122  # dimensionless
    mnu = 0.11  # eV
    As = 2.0e-9  # dimensionless
    ns = 0.965  # dimensionless
    omk = 0.0  # flat universe

    # k grid (h/Mpc)
    kh_min = 1e-4  # h/Mpc
    kh_max = 2.0   # h/Mpc
    n_k = 200
    kh_grid = np.linspace(kh_min, kh_max, n_k)

    # kmax for CAMB internal calculation (Mpc^-1)
    h_param = H0 / 100.0  # dimensionless
    kmax_calc = 5.0  # Mpc^-1, covers kh_max/h

    # Compute P(k) for both hierarchies
    pk_normal = compute_pk(
        H0, ombh2, omch2, mnu, As, ns, omk, "normal", kmax_calc, kh_grid
    )
    pk_inverted = compute_pk(
        H0, ombh2, omch2, mnu, As, ns, omk, "inverted", kmax_calc, kh_grid
    )

    # Relative difference
    rel_diff = (pk_inverted / pk_normal) - 1.0

    # Save to CSV
    df = pd.DataFrame({"k": kh_grid, "rel_diff": rel_diff})
    output_path = os.path.join("data", "result.csv")
    df.to_csv(output_path, index=False)

    # Print summary
    print("Relative difference in linear matter power spectrum (z=0) between inverted and normal neutrino hierarchies computed.")
    print("Cosmological parameters:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  Sum m_nu = " + str(mnu) + " eV")
    print("  A_s = " + str(As))
    print("  n_s = " + str(ns))
    print("k range: " + str(kh_min) + " < k < " + str(kh_max) + " h/Mpc, " + str(n_k) + " points")
    print("Results saved to " + output_path)
    print("First 5 rows of results:")
    print(df.head())


if __name__ == "__main__":
    main()
