# filename: codebase/camb_power_spectrum.py
import numpy as np
import pandas as pd
import os
from camb import model, initialpower, get_matter_power_interpolator

def compute_pk(cosmology_params, hierarchy, k_h, z=0.0):
    r"""
    Compute the linear matter power spectrum P(k) at redshift z for a given neutrino hierarchy.

    Parameters
    ----------
    cosmology_params : dict
        Dictionary containing cosmological parameters:
            - H0: Hubble constant in km/s/Mpc
            - ombh2: Baryon density parameter (Omega_b h^2)
            - omch2: Cold dark matter density parameter (Omega_c h^2)
            - mnu: Sum of neutrino masses in eV
            - As: Scalar amplitude
            - ns: Scalar spectral index
    hierarchy : str
        Neutrino mass hierarchy, either 'normal' or 'inverted'.
    k_h : array_like
        Array of wavenumbers in units of h/Mpc.
    z : float, optional
        Redshift at which to compute the power spectrum (default is 0.0).

    Returns
    -------
    pk : ndarray
        Linear matter power spectrum P(k) in units of (Mpc/h)^3 at the specified k_h and z.
    """
    # Set up CAMB parameters
    pars = model.CAMBparams()
    pars.set_cosmology(H0=cosmology_params['H0'],
                       ombh2=cosmology_params['ombh2'],
                       omch2=cosmology_params['omch2'],
                       mnu=cosmology_params['mnu'],
                       nnu=3.046)
    pars.InitPower.set_params(As=cosmology_params['As'], ns=cosmology_params['ns'])
    pars.set_matter_power(redshifts=[z], kmax=np.max(k_h)*1.1)
    # Set neutrino hierarchy
    if hierarchy == 'normal':
        pars.set_neutrino_hierarchy('normal')
    elif hierarchy == 'inverted':
        pars.set_neutrino_hierarchy('inverted')
    else:
        raise ValueError("hierarchy must be 'normal' or 'inverted'")
    # Get matter power interpolator
    PK = get_matter_power_interpolator(pars, nonlinear=False, hubble_units=False, k_hunit=True, kmax=np.max(k_h)*1.1)
    pk = PK.P(z, k_h)
    return pk

def main():
    r"""
    Main function to compute and save the relative difference in the linear matter power spectrum
    between inverted and normal neutrino hierarchies for the specified cosmology.
    """
    # Cosmological parameters (units in comments)
    cosmology_params = {
        'H0': 67.5,         # Hubble constant [km/s/Mpc]
        'ombh2': 0.022,     # Omega_b h^2 [dimensionless]
        'omch2': 0.122,     # Omega_c h^2 [dimensionless]
        'mnu': 0.11,        # Sum of neutrino masses [eV]
        'As': 2e-9,         # Scalar amplitude [dimensionless]
        'ns': 0.965         # Scalar spectral index [dimensionless]
    }
    # k range: 200 points from 1e-4 to 2 (h/Mpc)
    k_h = np.linspace(1e-4, 2.0, 200)
    # Compute P(k) for both hierarchies
    pk_normal = compute_pk(cosmology_params, 'normal', k_h, z=0.0)
    pk_inverted = compute_pk(cosmology_params, 'inverted', k_h, z=0.0)
    # Relative difference
    rel_diff = pk_inverted / pk_normal - 1.0
    # Prepare results
    df = pd.DataFrame({'k': k_h, 'rel_diff': rel_diff})
    # Ensure data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # Save to CSV
    csv_path = os.path.join(data_dir, "result.csv")
    df.to_csv(csv_path, index=False)
    # Print summary
    print("Relative difference in linear matter power spectrum (P_inverted / P_normal - 1) at z=0")
    print("Cosmological parameters:")
    for key, val in cosmology_params.items():
        print("  " + str(key) + ": " + str(val))
    print("k range: " + str(k_h[0]) + " to " + str(k_h[-1]) + " h/Mpc, total " + str(len(k_h)) + " points")
    print("Results saved to: " + csv_path)
    print("First 5 rows of the result:")
    print(df.head())
    print("Last 5 rows of the result:")
    print(df.tail())

if __name__ == "__main__":
    main()
