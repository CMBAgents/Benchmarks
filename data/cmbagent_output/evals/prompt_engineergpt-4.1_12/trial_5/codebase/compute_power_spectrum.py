# filename: codebase/compute_power_spectrum.py
r"""
Compute the relative difference in the linear matter power spectrum P(k) at z=0
between normal and inverted neutrino hierarchies using CAMB, for a flat Lambda CDM cosmology.

Results are saved in data/result.csv with columns:
- k: Wavenumber in h/Mpc
- rel_diff: (P_inverted / P_normal - 1)

Units:
- k: h/Mpc
- P(k): (Mpc/h)^3

Requirements:
- camb

Author: Engineer Agent
"""

import numpy as np
import pandas as pd
import os
import camb
from camb import model


def compute_pk(cosmo_params, k_h, hierarchy):
    r"""
    Compute the linear matter power spectrum P(k) at z=0 for given cosmological parameters and neutrino hierarchy.

    Parameters
    ----------
    cosmo_params : dict
        Dictionary with cosmological parameters.
    k_h : ndarray
        Array of wavenumbers in h/Mpc.
    hierarchy : str
        'normal' or 'inverted' for neutrino mass hierarchy.

    Returns
    -------
    pk : ndarray
        Linear matter power spectrum at z=0, units (Mpc/h)^3, same shape as k_h.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=cosmo_params['H0'],
                       ombh2=cosmo_params['ombh2'],
                       omch2=cosmo_params['omch2'],
                       mnu=cosmo_params['mnu'],
                       nnu=3.046,
                       neutrino_hierarchy=hierarchy)
    pars.InitPower.set_params(As=cosmo_params['As'], ns=cosmo_params['ns'])
    pars.set_matter_power(redshifts=[0.0], kmax=np.max(k_h)*1.05)
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=np.min(k_h), maxkh=np.max(k_h), npoints=len(k_h))
    # Interpolate to requested k_h grid
    pk_interp = np.interp(k_h, kh, pk[0])
    return pk_interp


def main():
    r"""
    Main routine to compute and save the relative difference in the linear matter power spectrum
    between inverted and normal neutrino hierarchies.
    """
    # Cosmological parameters
    cosmo_params = {
        'H0': 67.5,         # Hubble constant [km/s/Mpc]
        'ombh2': 0.022,     # Omega_b h^2
        'omch2': 0.122,     # Omega_c h^2
        'mnu': 0.11,        # sum m_nu [eV]
        'As': 2e-9,         # Scalar amplitude
        'ns': 0.965         # Scalar spectral index
    }

    # k grid: 200 points, 1e-4 < k < 2 (h/Mpc)
    k_h = np.linspace(1e-4, 2.0, 200)

    # Compute P(k) for both hierarchies
    pk_normal = compute_pk(cosmo_params, k_h, hierarchy='normal')
    pk_inverted = compute_pk(cosmo_params, k_h, hierarchy='inverted')

    # Relative difference
    rel_diff = pk_inverted / pk_normal - 1.0

    # Prepare output directory
    out_dir = "data"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Save to CSV
    df = pd.DataFrame({'k': k_h, 'rel_diff': rel_diff})
    out_file = os.path.join(out_dir, "result.csv")
    df.to_csv(out_file, index=False)

    # Print summary
    np.set_printoptions(precision=6, suppress=True)
    print("Relative difference in linear matter power spectrum (P_inverted / P_normal - 1) at z=0")
    print("Cosmological parameters:")
    for k, v in cosmo_params.items():
        print("  " + str(k) + ": " + str(v))
    print("k range: " + str(k_h[0]) + " to " + str(k_h[-1]) + " h/Mpc, total " + str(len(k_h)) + " points")
    print("Saved results to " + out_file)
    print("First 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()