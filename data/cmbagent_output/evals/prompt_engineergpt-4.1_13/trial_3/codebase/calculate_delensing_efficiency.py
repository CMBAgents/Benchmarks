# filename: codebase/calculate_delensing_efficiency.py
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
    l = df['l'].values
    Nl = df['Nl'].values
    # Only keep l <= lmax
    mask = (l >= 2) & (l <= lmax)
    l = l[mask]
    Nl = Nl[mask]
    # Convert Nl to N0: Nl = N0 * (l(l+1))^2 / (2pi) => N0 = Nl * (2pi) / (l(l+1))^2
    factor = (2.0 * np.pi) / (l * (l + 1))**2
    N0 = Nl * factor
    return l, N0

def get_camb_results(lmax, params_dict):
    r"""
    Set up CAMB with the given cosmological parameters and compute the lensed B-mode and lensing potential power spectra.

    Parameters
    ----------
    lmax : int
        Maximum multipole to compute.
    params_dict : dict
        Dictionary of cosmological parameters.

    Returns
    -------
    ell : ndarray
        Multipole moments (int) from 2 to lmax.
    cl_bb_lensed : ndarray
        Lensed B-mode power spectrum (float), in microK^2.
    cl_pp : ndarray
        Lensing potential power spectrum (float), in C_ell^{\phi\phi} units.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=params_dict['H0'],
                       ombh2=params_dict['ombh2'],
                       omch2=params_dict['omch2'],
                       mnu=params_dict['mnu'],
                       omk=params_dict['omk'],
                       tau=params_dict['tau'])
    pars.InitPower.set_params(As=params_dict['As'], ns=params_dict['ns'])
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.WantTensors = False
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total', 'lensed_scalar'])
    cl_bb_lensed = powers['total'][:, 2]  # BB is index 2
    # Get lensing potential power spectrum
    cl_pp = results.get_lens_potential_cls(lmax=lmax)[:, 0]  # phi-phi
    ell = np.arange(cl_pp.shape[0])
    return ell, cl_bb_lensed, cl_pp

def compute_residual_clpp(cl_pp, n0):
    r"""
    Compute the residual lensing potential power spectrum.

    Parameters
    ----------
    cl_pp : ndarray
        Lensing potential power spectrum (float).
    n0 : ndarray
        Lensing noise power spectrum (float).

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
        Zero-padded array of length 'length'.
    """
    arr_padded = np.zeros(length)
    arr_padded[:min(len(arr), length)] = arr[:min(len(arr), length)]
    return arr_padded

def compute_delensed_bb(lmax, params_dict, cl_pp_res):
    r"""
    Compute the delensed B-mode power spectrum using the residual lensing potential.

    Parameters
    ----------
    lmax : int
        Maximum multipole to compute.
    params_dict : dict
        Dictionary of cosmological parameters.
    cl_pp_res : ndarray
        Residual lensing potential power spectrum (float).

    Returns
    -------
    cl_bb_delensed : ndarray
        Delensed B-mode power spectrum (float), in microK^2.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=params_dict['H0'],
                       ombh2=params_dict['ombh2'],
                       omch2=params_dict['omch2'],
                       mnu=params_dict['mnu'],
                       omk=params_dict['omk'],
                       tau=params_dict['tau'])
    pars.InitPower.set_params(As=params_dict['As'], ns=params_dict['ns'])
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.WantTensors = False
    # Set custom lensing potential
    pars.set_custom_lens_potential(cl_pp_res)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total', 'lensed_scalar'])
    cl_bb_delensed = powers['total'][:, 2]  # BB is index 2
    return cl_bb_delensed

def main():
    r"""
    Main function to compute the delensing efficiency of the CMB B-mode polarization power spectrum.
    Saves the results in 'data/result.csv' and prints detailed results to the console.
    """
    # Cosmological parameters
    params_dict = {
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
    n0_filepath = '/Users/antoidicherianlonappan/Software/Benchmarks/data/N0.csv'
    # Load N0
    l_n0, n0 = load_n0_csv(n0_filepath, lmax)
    # Get CAMB results
    ell, cl_bb_lensed, cl_pp = get_camb_results(lmax, params_dict)
    # Truncate or pad cl_pp and n0 to length lmax+1 (CAMB returns 0..lmax)
    cl_pp = pad_to_length(cl_pp, lmax + 1)
    n0_full = np.zeros(lmax + 1)
    n0_full[l_n0] = n0
    # Compute residual lensing potential
    cl_pp_res = compute_residual_clpp(cl_pp, n0_full)
    # Pad residual to CAMB's required length
    cl_pp_res_padded = pad_to_length(cl_pp_res, lmax + 1)
    # Compute delensed BB
    cl_bb_delensed = compute_delensed_bb(lmax, params_dict, cl_pp_res_padded)
    # Compute delensing efficiency for l=2 to l=100
    l_range = np.arange(2, 101)
    cl_bb_lensed_sel = cl_bb_lensed[l_range]
    cl_bb_delensed_sel = cl_bb_delensed[l_range]
    delensing_efficiency = 100.0 * (cl_bb_lensed_sel - cl_bb_delensed_sel) / cl_bb_lensed_sel
    # Save results
    result_df = pd.DataFrame({'l': l_range, 'delensing_efficiency': delensing_efficiency})
    result_df.to_csv('data/result.csv', index=False)
    # Print results
    np.set_printoptions(precision=6, suppress=True)
    print("Delensing efficiency (in percent) for l=2 to l=100:")
    print(result_df)
    print("\nSaved delensing efficiency results to data/result.csv")
    # Print summary statistics
    print("\nSummary statistics for delensing efficiency (l=2 to l=100):")
    print("Mean: " + str(np.mean(delensing_efficiency)))
    print("Min: " + str(np.min(delensing_efficiency)))
    print("Max: " + str(np.max(delensing_efficiency)))

if __name__ == "__main__":
    main()