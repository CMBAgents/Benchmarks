# filename: codebase/neutrino_matter_power.py
import camb
from camb import model
import numpy as np
import pandas as pd
import os


def compute_pk(
    H0,
    ombh2,
    omch2,
    mnu,
    As,
    ns,
    omk,
    neutrino_hierarchy,
    kh_grid,
    kmax_calc,
    z=0.0
):
    r"""
    Compute the linear matter power spectrum P(k) at redshift z for given cosmological parameters
    and neutrino hierarchy using CAMB.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter (Omega_b h^2).
    omch2 : float
        Physical cold dark matter density parameter (Omega_c h^2).
    mnu : float
        Sum of neutrino masses in eV.
    As : float
        Scalar amplitude at k_pivot=0.05 Mpc^-1.
    ns : float
        Scalar spectral index.
    omk : float
        Curvature parameter (0 for flat).
    neutrino_hierarchy : str
        'normal' or 'inverted'.
    kh_grid : np.ndarray
        Array of k values in h/Mpc at which to evaluate P(k).
    kmax_calc : float
        Maximum k (in Mpc^-1) for CAMB's internal calculation.
    z : float
        Redshift at which to compute P(k).

    Returns
    -------
    pk : np.ndarray
        Array of P(k) values in (Mpc/h)^3 at the specified kh_grid and redshift.
    """
    pars = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu,
        omk=omk,
        As=As,
        ns=ns,
        neutrino_hierarchy=neutrino_hierarchy,
        WantScalars=True,
        WantTensors=False,
        WantVectors=False,
        WantDerivedParameters=True,
        WantTransfer=True
    )
    pars.NonLinear = model.NonLinear_none
    pars.set_matter_power(redshifts=[z], kmax=kmax_calc)
    results = camb.get_results(pars)
    PK_interp = results.get_matter_power_interpolator(
        nonlinear=False,
        var1='delta_tot',
        var2='delta_tot',
        hubble_units=True,
        k_hunit=True,
        log_interp=True
    )
    pk = PK_interp.P(z, kh_grid)
    return pk


def main():
    r"""
    Main routine to compute and save the relative difference in the linear matter power spectrum
    between inverted and normal neutrino hierarchies.
    """
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
    kh_grid = np.linspace(kh_min, kh_max, n_k)  # h/Mpc

    # kmax for CAMB's internal calculation (Mpc^-1)
    h_param = H0 / 100.0
    kmax_calc = 5.0  # Mpc^-1, safely above kh_max/h

    # Compute P(k) for normal hierarchy
    pk_normal = compute_pk(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu,
        As=As,
        ns=ns,
        omk=omk,
        neutrino_hierarchy='normal',
        kh_grid=kh_grid,
        kmax_calc=kmax_calc,
        z=0.0
    )

    # Compute P(k) for inverted hierarchy
    pk_inverted = compute_pk(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu,
        As=As,
        ns=ns,
        omk=omk,
        neutrino_hierarchy='inverted',
        kh_grid=kh_grid,
        kmax_calc=kmax_calc,
        z=0.0
    )

    # Relative difference
    rel_diff = (pk_inverted / pk_normal) - 1.0

    # Prepare output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({
        "k": kh_grid,  # h/Mpc
        "rel_diff": rel_diff
    })
    df.to_csv(output_path, index=False)

    # Print summary
    print("Relative difference in linear matter power spectrum (inverted vs normal hierarchy) computed.")
    print("k (h/Mpc) and rel_diff (dimensionless) saved to " + output_path)
    print("First 5 rows of the result:")
    print(df.head())


if __name__ == "__main__":
    main()