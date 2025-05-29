# filename: codebase/camb_power_spectrum.py
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower
import os

def compute_pk(cosmo_params, k_h, hierarchy):
    r"""
    Compute the linear matter power spectrum P(k) at z=0 for a given neutrino hierarchy.

    Parameters
    ----------
    cosmo_params : dict
        Dictionary containing cosmological parameters:
            - H0: Hubble constant in km/s/Mpc
            - ombh2: Baryon density parameter (Omega_b h^2)
            - omch2: Cold dark matter density parameter (Omega_c h^2)
            - mnu: Sum of neutrino masses in eV
            - As: Scalar amplitude
            - ns: Scalar spectral index
    k_h : ndarray
        Array of wavenumbers in units of h/Mpc.
    hierarchy : str
        Neutrino mass hierarchy, either 'normal' or 'inverted'.

    Returns
    -------
    pk : ndarray
        Linear matter power spectrum P(k) in (Mpc/h)^3 at z=0 for the given k_h.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=cosmo_params['H0'],
                       ombh2=cosmo_params['ombh2'],
                       omch2=cosmo_params['omch2'],
                       mnu=cosmo_params['mnu'],
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
    Main function to compute and save the relative difference in the linear matter power spectrum
    between inverted and normal neutrino hierarchies for specified cosmological parameters.
    """
    # Cosmological parameters (units in comments)
    cosmo_params = {
        'H0': 67.5,         # Hubble constant [km/s/Mpc]
        'ombh2': 0.022,     # Omega_b h^2 [dimensionless]
        'omch2': 0.122,     # Omega_c h^2 [dimensionless]
        'mnu': 0.11,        # Sum of neutrino masses [eV]
        'As': 2e-9,         # Scalar amplitude [dimensionless]
        'ns': 0.965         # Scalar spectral index [dimensionless]
    }

    # k in h/Mpc, 200 points from 1e-4 to 2
    k_h = np.linspace(1e-4, 2.0, 200)

    # Compute P(k) for both hierarchies
    pk_normal = compute_pk(cosmo_params, k_h, hierarchy='normal')
    pk_inverted = compute_pk(cosmo_params, k_h, hierarchy='inverted')

    # Relative difference
    rel_diff = pk_inverted / pk_normal - 1.0

    # Prepare results
    df = pd.DataFrame({'k': k_h, 'rel_diff': rel_diff})

    # Ensure data directory exists
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save to CSV
    csv_path = os.path.join(data_dir, 'result.csv')
    df.to_csv(csv_path, index=False)

    # Print summary
    np.set_printoptions(precision=6, suppress=True)
    print("Relative difference in linear matter power spectrum (P_inverted / P_normal - 1) at z=0")
    print("Cosmological parameters:")
    for key, value in cosmo_params.items():
        print("  " + str(key) + ": " + str(value))
    print("k range: " + str(k_h[0]) + " to " + str(k_h[-1]) + " h/Mpc, total points: " + str(len(k_h)))
    print("First 5 rows of results:")
    print(df.head())
    print("Results saved to: " + csv_path)


if __name__ == "__main__":
    main()