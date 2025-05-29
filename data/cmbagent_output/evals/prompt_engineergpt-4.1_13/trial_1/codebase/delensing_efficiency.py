# filename: codebase/delensing_efficiency.py
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

def load_n0_csv(filepath, lmax):
    r"""
    Load the lensing noise power spectrum N0 from a CSV file and convert to N0 units.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing columns 'l' and 'Nl'.
    lmax : int
        Maximum multipole to consider.

    Returns
    -------
    l_arr : ndarray
        Array of multipole moments (int) up to lmax.
    n0_arr : ndarray
        Array of N0 values (float) up to lmax, in units of C_ell^{\phi\phi}.
    """
    df = pd.read_csv(filepath)
    l = df['l'].values.astype(int)
    Nl = df['Nl'].values
    # Only keep l <= lmax
    mask = (l >= 2) & (l <= lmax)
    l = l[mask]
    Nl = Nl[mask]
    # Convert Nl to N0: Nl = N0 * (l(l+1))^2 / (2pi) => N0 = Nl * (2pi) / (l(l+1))^2
    factor = (2.0 * np.pi) / ((l * (l + 1)) ** 2)
    N0 = Nl * factor
    return l, N0

def camb_get_cls(params, lmax, clpp_override=None):
    r"""
    Run CAMB to get lensed BB and lensing potential power spectra.

    Parameters
    ----------
    params : dict
        Dictionary of cosmological parameters.
    lmax : int
        Maximum multipole.
    clpp_override : ndarray or None
        If provided, overrides the lensing potential power spectrum with this array.

    Returns
    -------
    l : ndarray
        Multipole moments (int) from 2 to lmax.
    cl_bb : ndarray
        Lensed BB power spectrum (float), in microK^2.
    cl_pp : ndarray
        Lensing potential power spectrum (float), in C_ell^{\phi\phi}.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=params['H0'], ombh2=params['ombh2'], omch2=params['omch2'],
                       mnu=params['mnu'], omk=params['omk'], tau=params['tau'])
    pars.InitPower.set_params(As=params['As'], ns=params['ns'])
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.WantTensors = False
    pars.Want_CMB_lensing = True

    if clpp_override is not None:
        # Use the residual lensing potential for delensed BB
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True, lens_potential_input=clpp_override)
    else:
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)

    # lensed BB
    cl = powers['lensed_scalar']
    l = np.arange(cl.shape[0])
    cl_bb = cl[:, 2]  # BB
    # lensing potential
    cl_pp = results.get_lens_potential_cls(lmax=lmax)[:, 0]  # C_ell^{\phi\phi}
    return l, cl_bb, cl_pp

def compute_residual_clpp(cl_pp, n0):
    r"""
    Compute the residual lensing potential power spectrum.

    Parameters
    ----------
    cl_pp : ndarray
        Lensing potential power spectrum (float), C_ell^{\phi\phi}.
    n0 : ndarray
        Lensing noise power spectrum (float), N0.

    Returns
    -------
    cl_pp_res : ndarray
        Residual lensing potential power spectrum (float).
    """
    # Avoid division by zero
    denom = cl_pp + n0
    mask = denom > 0
    cl_pp_res = np.zeros_like(cl_pp)
    cl_pp_res[mask] = cl_pp[mask] * (1.0 - (cl_pp[mask] / denom[mask]))
    return cl_pp_res

def pad_to_length(arr, length):
    r"""
    Pad an array with zeros to the specified length.

    Parameters
    ----------
    arr : ndarray
        Input array.
    length : int
        Desired length.

    Returns
    -------
    arr_padded : ndarray
        Padded array.
    """
    arr_padded = np.zeros(length)
    arr_padded[:min(len(arr), length)] = arr[:min(len(arr), length)]
    return arr_padded

def main():
    r"""
    Main function to compute the delensing efficiency of the CMB B-mode polarization power spectrum.
    """
    # Cosmological parameters
    params = {
        'H0': 67.5,         # Hubble constant [km/s/Mpc]
        'ombh2': 0.022,     # Baryon density
        'omch2': 0.122,     # Cold dark matter density
        'mnu': 0.06,        # Neutrino mass sum [eV]
        'omk': 0.0,         # Curvature
        'tau': 0.06,        # Optical depth
        'As': 2e-9,         # Scalar amplitude
        'ns': 0.965         # Scalar spectral index
    }
    lmax = 2000

    # Load N0
    n0_filepath = '/Users/antoidicherianlonappan/Software/Benchmarks/data/N0.csv'
    l_n0, n0 = load_n0_csv(n0_filepath, lmax)
    # Pad N0 to lmax+1 (CAMB arrays go from l=0)
    n0_full = np.zeros(lmax + 1)
    n0_full[l_n0] = n0

    # Get lensed BB and lensing potential
    l, cl_bb, cl_pp = camb_get_cls(params, lmax)
    # Ensure cl_pp and n0_full have same length
    cl_pp = pad_to_length(cl_pp, lmax + 1)
    n0_full = pad_to_length(n0_full, lmax + 1)

    # Compute residual lensing potential
    cl_pp_res = compute_residual_clpp(cl_pp, n0_full)

    # Pad residual lensing potential to CAMB's required length
    cl_pp_res_padded = pad_to_length(cl_pp_res, lmax + 1)

    # Compute delensed BB using CAMB with residual lensing potential
    # Note: CAMB expects the full array for lens_potential_input
    l2, cl_bb_delensed, _ = camb_get_cls(params, lmax, clpp_override=cl_pp_res_padded)

    # Compute delensing efficiency for l=2 to l=100
    l_range = np.arange(2, 101)
    cl_bb_lensed = cl_bb[l_range]
    cl_bb_delensed = cl_bb_delensed[l_range]
    delensing_efficiency = 100.0 * (cl_bb_lensed - cl_bb_delensed) / cl_bb_lensed

    # Save results to CSV
    result_df = pd.DataFrame({
        'l': l_range,
        'delensing_efficiency': delensing_efficiency
    })
    result_path = 'data/result.csv'
    result_df.to_csv(result_path, index=False)

    # Print results
    np.set_printoptions(precision=6, suppress=True)
    pd.set_option('display.max_rows', 100)
    print("\nDelensing efficiency (percent) for l=2 to l=100:")
    print(result_df)
    print("\nResults saved to " + result_path)


if __name__ == "__main__":
    main()
