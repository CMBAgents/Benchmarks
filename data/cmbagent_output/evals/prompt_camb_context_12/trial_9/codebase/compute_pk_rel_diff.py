# filename: codebase/compute_pk_rel_diff.py
import camb
from camb import model
import numpy as np
import pandas as pd
import os


def compute_pk(hierarchy, H0, ombh2, omch2, mnu_sum, As, ns, tau, kmax, kh_vals):
    r"""
    Compute the linear matter power spectrum P(k) at z=0 for a given neutrino hierarchy.

    Parameters
    ----------
    hierarchy : str
        Neutrino hierarchy, either 'normal' or 'inverted'.
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter Ω_b h^2 (dimensionless).
    omch2 : float
        Physical cold dark matter density parameter Ω_c h^2 (dimensionless).
    mnu_sum : float
        Sum of neutrino masses in eV.
    As : float
        Scalar amplitude of primordial power spectrum (dimensionless).
    ns : float
        Scalar spectral index (dimensionless).
    tau : float
        Optical depth to reionization (dimensionless).
    kmax : float
        Maximum wavenumber in Mpc^-1 for CAMB calculation.
    kh_vals : np.ndarray
        Array of wavenumbers in h/Mpc at which to evaluate P(k).

    Returns
    -------
    pk : np.ndarray
        Linear matter power spectrum at z=0, in units of (Mpc/h)^3.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu_sum, omk=0.0,
                       neutrino_hierarchy=hierarchy, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=[0.0], kmax=kmax)
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    PK = results.get_matter_power_interpolator(
        nonlinear=False,
        var1='delta_tot',
        var2='delta_tot',
        hubble_units=True,
        k_hunit=True
    )
    pk = PK.P(0.0, kh_vals)
    return pk


def main():
    r"""
    Main routine to compute and save the relative difference in the linear matter power spectrum
    between inverted and normal neutrino hierarchies at z=0 for specified cosmological parameters.

    Saves results to 'data/result.csv' with columns:
    - k: Wavenumber in h/Mpc (float, 200 values)
    - rel_diff: Relative difference (P_inverted/P_normal - 1) (float)
    """
    # Cosmological parameters (units in comments)
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density Ω_b h^2 [dimensionless]
    omch2 = 0.122  # CDM density Ω_c h^2 [dimensionless]
    mnu_sum = 0.11  # Neutrino mass sum [eV]
    As = 2e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]
    tau = 0.06  # Optical depth [dimensionless]

    h = H0 / 100.0  # Dimensionless Hubble parameter

    # Wavenumber array: 200 evenly spaced values in 1e-4 < k < 2 (h/Mpc)
    kh_vals = np.linspace(1e-4, 2.0, 200)  # [h/Mpc]

    # kmax for CAMB (in Mpc^-1): must be > max(k/h)
    kmax = (kh_vals.max() / h) * 1.05  # [Mpc^-1]

    # Compute P(k) for both hierarchies
    print("Computing linear matter power spectrum for normal hierarchy..." )
    pk_normal = compute_pk('normal', H0, ombh2, omch2, mnu_sum, As, ns, tau, kmax, kh_vals)
    print("Computing linear matter power spectrum for inverted hierarchy..." )
    pk_inverted = compute_pk('inverted', H0, ombh2, omch2, mnu_sum, As, ns, tau, kmax, kh_vals)

    # Compute relative difference
    rel_diff = (pk_inverted / pk_normal) - 1

    # Prepare output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({
        "k": kh_vals,  # [h/Mpc]
        "rel_diff": rel_diff
    })
    df.to_csv(output_path, index=False)
    print("Relative difference in linear matter power spectrum saved to " + output_path)
    print("First 5 rows of the result:")
    print(df.head())


if __name__ == "__main__":
    main()