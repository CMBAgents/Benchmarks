# filename: codebase/compute_power_spectrum.py
import numpy as np
import pandas as pd
import camb
import os

def compute_pk(cosmology_params, k_h, hierarchy):
    r"""
    Compute the linear matter power spectrum P(k) at z=0 for given cosmology and neutrino hierarchy.

    Parameters
    ----------
    cosmology_params : dict
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
    pars.set_cosmology(H0=cosmology_params['H0'],
                       ombh2=cosmology_params['ombh2'],
                       omch2=cosmology_params['omch2'],
                       mnu=cosmology_params['mnu'],
                       neutrino_hierarchy=hierarchy)
    pars.InitPower.set_params(As=cosmology_params['As'], ns=cosmology_params['ns'])
    pars.set_matter_power(redshifts=[0.0], kmax=np.max(k_h)*1.05)
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=np.min(k_h), maxkh=np.max(k_h), npoints=len(k_h))
    # Interpolate to requested k_h
    pk_interp = np.interp(k_h, kh, pk[0])
    return pk_interp

def main():
    r"""
    Main routine to compute and save the relative difference in P(k) between inverted and normal hierarchies.
    """
    # Cosmological parameters
    params = {
        'H0': 67.5,         # Hubble constant [km/s/Mpc]
        'ombh2': 0.022,     # Omega_b h^2
        'omch2': 0.122,     # Omega_c h^2
        'mnu': 0.11,        # Sum of neutrino masses [eV]
        'As': 2e-9,         # Scalar amplitude
        'ns': 0.965         # Scalar spectral index
    }

    # k range and sampling
    kmin = 1e-4  # h/Mpc
    kmax = 2.0   # h/Mpc
    n_k = 200
    k_h = np.linspace(kmin, kmax, n_k)

    # Compute P(k) for both hierarchies
    pk_normal = compute_pk(params, k_h, hierarchy='normal')
    pk_inverted = compute_pk(params, k_h, hierarchy='inverted')

    # Relative difference
    rel_diff = pk_inverted / pk_normal - 1

    # Save to CSV
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'k': k_h, 'rel_diff': rel_diff})
    df.to_csv(output_file, index=False)

    # Print summary
    np.set_printoptions(precision=6, suppress=True)
    print("Relative difference in linear matter power spectrum (P_inverted / P_normal - 1) at z=0")
    print("Cosmological parameters:")
    for key in params:
        print("  " + key + ": " + str(params[key]))
    print("k range: " + str(kmin) + " to " + str(kmax) + " h/Mpc, n_k = " + str(n_k))
    print("Saved results to: " + output_file)
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())


if __name__ == "__main__":
    main()
