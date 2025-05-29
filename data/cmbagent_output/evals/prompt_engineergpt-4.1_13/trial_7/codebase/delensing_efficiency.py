# filename: codebase/delensing_efficiency.py
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

def load_n0_csv(filepath, lmax):
    r"""
    Load the lensing noise power spectrum N0 from a CSV file and return arrays up to lmax.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file containing columns 'l' and 'Nl'.
    lmax : int
        Maximum multipole to include.
    
    Returns
    -------
    l_arr : ndarray
        Array of multipole moments (int) up to lmax.
    n0_arr : ndarray
        Array of N0 values (float) up to lmax, in units of C_ell^{\phi\phi}.
    """
    df = pd.read_csv(filepath)
    l_arr = df['l'].values.astype(int)
    Nl = df['Nl'].values
    # Nl = N0 * (l(l+1))^2 / (2pi) => N0 = Nl * (2pi) / (l(l+1))^2
    n0 = np.zeros(lmax+1)
    for i, ell in enumerate(l_arr):
        if ell > lmax:
            break
        if ell == 0:
            continue
        n0[ell] = Nl[i] * (2.0 * np.pi) / (ell * (ell + 1))**2
    return np.arange(lmax+1), n0

def get_camb_results(lmax, params_dict):
    r"""
    Set up CAMB with the given cosmological parameters and return the lensed BB and lensing potential spectra.
    
    Parameters
    ----------
    lmax : int
        Maximum multipole to compute.
    params_dict : dict
        Dictionary of cosmological parameters.
    
    Returns
    -------
    l_arr : ndarray
        Array of multipole moments (int) up to lmax.
    cl_bb_lensed : ndarray
        Lensed B-mode power spectrum (muK^2).
    cl_pp : ndarray
        Lensing potential power spectrum C_ell^{\phi\phi}.
    cl_pp_plot : ndarray
        Lensing potential power spectrum for plotting: C_ell^{\phi\phi} * (l(l+1))^2 / (2pi).
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=params_dict['H0'], ombh2=params_dict['ombh2'], omch2=params_dict['omch2'],
                       mnu=params_dict['mnu'], omk=params_dict['omk'], tau=params_dict['tau'])
    pars.InitPower.set_params(As=params_dict['As'], ns=params_dict['ns'])
    pars.set_for_lmax(lmax, lens_potential_accuracy=2)
    pars.WantTensors = False
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    cl_bb_lensed = powers['lensed_scalar'][:, 2]  # BB
    # Lensing potential power spectrum
    cl_pp = results.get_lens_potential_cls(lmax=lmax)[:, 0]  # phi-phi
    # For plotting: C_ell^{\phi\phi} * (l(l+1))^2 / (2.0 * np.pi)
    ells = np.arange(cl_pp.size)
    cl_pp_plot = cl_pp * (ells * (ells + 1))**2 / (2.0 * np.pi)
    return ells, cl_bb_lensed, cl_pp, cl_pp_plot

def compute_residual_phi_phi(cl_pp, n0):
    r"""
    Compute the residual lensing potential power spectrum.
    
    Parameters
    ----------
    cl_pp : ndarray
        Lensing potential power spectrum C_ell^{\phi\phi}.
    n0 : ndarray
        Lensing noise power spectrum N0 (same units as cl_pp).
    
    Returns
    -------
    cl_pp_res : ndarray
        Residual lensing potential power spectrum.
    """
    # Avoid division by zero
    denom = cl_pp + n0
    mask = denom > 0
    cl_pp_res = np.zeros_like(cl_pp)
    cl_pp_res[mask] = cl_pp[mask] * (1.0 - (cl_pp[mask] / denom[mask]))
    return cl_pp_res

def pad_to_length(arr, length):
    r"""
    Pad or truncate an array to the specified length.
    
    Parameters
    ----------
    arr : ndarray
        Input array.
    length : int
        Desired length.
    
    Returns
    -------
    arr_out : ndarray
        Array of length 'length'.
    """
    arr_out = np.zeros(length)
    n = min(len(arr), length)
    arr_out[:n] = arr[:n]
    return arr_out

def compute_delensed_BB(lmax, params_dict, cl_pp_res):
    r"""
    Compute the delensed B-mode power spectrum using the residual lensing potential.
    
    Parameters
    ----------
    lmax : int
        Maximum multipole to compute.
    params_dict : dict
        Dictionary of cosmological parameters.
    cl_pp_res : ndarray
        Residual lensing potential power spectrum.
    
    Returns
    -------
    cl_bb_delensed : ndarray
        Delensed B-mode power spectrum (muK^2).
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=params_dict['H0'], ombh2=params_dict['ombh2'], omch2=params_dict['omch2'],
                       mnu=params_dict['mnu'], omk=params_dict['omk'], tau=params_dict['tau'])
    pars.InitPower.set_params(As=params_dict['As'], ns=params_dict['ns'])
    pars.set_for_lmax(lmax, lens_potential_accuracy=2)
    pars.WantTensors = False
    # Set custom lensing potential
    pars.set_custom_lens_potential(cl_pp_res)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    cl_bb_delensed = powers['lensed_scalar'][:, 2]  # BB
    return cl_bb_delensed

def main():
    r"""
    Main function to compute the delensing efficiency of the CMB B-mode power spectrum.
    Saves the results in 'data/result.csv' and prints key results to the console.
    """
    # Cosmological parameters
    params_dict = {
        'H0': 67.5,         # Hubble constant [km/s/Mpc]
        'ombh2': 0.022,     # Omega_b h^2
        'omch2': 0.122,     # Omega_c h^2
        'mnu': 0.06,        # sum m_nu [eV]
        'omk': 0.0,         # Omega_k
        'tau': 0.06,        # optical depth
        'As': 2e-9,         # scalar amplitude
        'ns': 0.965         # scalar spectral index
    }
    lmax = 2000
    n0_filepath = '/Users/antoidicherianlonappan/Software/Benchmarks/data/N0.csv'
    # Load N0
    l_arr_n0, n0 = load_n0_csv(n0_filepath, lmax)
    # Get CAMB results
    l_arr, cl_bb_lensed, cl_pp, cl_pp_plot = get_camb_results(lmax, params_dict)
    # Ensure arrays are same length
    n0 = pad_to_length(n0, lmax+1)
    cl_pp = pad_to_length(cl_pp, lmax+1)
    # Compute residual lensing potential
    cl_pp_res = compute_residual_phi_phi(cl_pp, n0)
    # Pad residual lensing potential to CAMB's required length
    cl_pp_res = pad_to_length(cl_pp_res, lmax+1)
    # Compute delensed BB
    cl_bb_delensed = compute_delensed_BB(lmax, params_dict, cl_pp_res)
    # Compute delensing efficiency for l=2 to l=100
    lmin_eff = 2
    lmax_eff = 100
    l_eff = np.arange(lmin_eff, lmax_eff+1)
    cl_bb_lensed_eff = cl_bb_lensed[lmin_eff:lmax_eff+1]
    cl_bb_delensed_eff = cl_bb_delensed[lmin_eff:lmax_eff+1]
    # Avoid division by zero
    mask = cl_bb_lensed_eff != 0
    delensing_efficiency = np.zeros_like(cl_bb_lensed_eff)
    delensing_efficiency[mask] = 100.0 * (cl_bb_lensed_eff[mask] - cl_bb_delensed_eff[mask]) / cl_bb_lensed_eff[mask]
    # Save results
    result_df = pd.DataFrame({'l': l_eff, 'delensing_efficiency': delensing_efficiency})
    result_df.to_csv('data/result.csv', index=False)
    # Print results
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print("Delensing efficiency (percent) for l=2 to l=100:")
    print(result_df)
    print("\nLensed BB power spectrum (muK^2) for l=2 to l=10:")
    print(cl_bb_lensed[2:11])
    print("\nDelensed BB power spectrum (muK^2) for l=2 to l=10:")
    print(cl_bb_delensed[2:11])
    print("\nLensing potential power spectrum (C_ell^{phi phi} * (l(l+1))^2 / 2pi) for l=2 to l=10:")
    print(cl_pp_plot[2:11])
    print("\nResidual lensing potential power spectrum (C_ell^{phi phi, res}) for l=2 to l=10:")
    print(cl_pp_res[2:11])
    print("\nResults saved to data/result.csv")


if __name__ == "__main__":
    main()