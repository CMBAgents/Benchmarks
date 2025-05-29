# filename: codebase/delensing_efficiency.py
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

def load_n0_csv(filepath, lmax):
    r"""
    Load the lensing noise power spectrum N0 from a CSV file and process it up to lmax.

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
        Array of N0 values up to lmax, in units of (dimensionless, as in input).
    """
    df = pd.read_csv(filepath)
    df = df[df['l'] <= lmax]
    l_arr = df['l'].values.astype(int)
    n0_arr = df['Nl'].values
    # Ensure l runs from 0 to lmax
    n0_full = np.zeros(lmax+1)
    n0_full[l_arr] = n0_arr
    return np.arange(lmax+1), n0_full

def camb_get_cls_and_phi(cosmo_params, lmax):
    r"""
    Run CAMB to get lensed BB and lensing potential power spectra up to lmax.

    Parameters
    ----------
    cosmo_params : dict
        Dictionary of cosmological parameters.
    lmax : int
        Maximum multipole.

    Returns
    -------
    l_arr : ndarray
        Array of multipole moments (int) up to lmax.
    cl_bb : ndarray
        Lensed BB power spectrum (muK^2), length lmax+1.
    cl_pp : ndarray
        Lensing potential power spectrum (dimensionless), length lmax+1.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=cosmo_params['H0'],
                       ombh2=cosmo_params['ombh2'],
                       omch2=cosmo_params['omch2'],
                       mnu=cosmo_params['mnu'],
                       omk=cosmo_params['omk'],
                       tau=cosmo_params['tau'])
    pars.InitPower.set_params(As=cosmo_params['As'], ns=cosmo_params['ns'])
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    lensed = powers['total']
    # lensed[:,2] is BB, lensed[:,4] is phi-phi (CAMB returns phi-phi as 4th column)
    # But get_lens_potential_cls is more direct for phi-phi
    cl_bb = lensed[:,2]  # muK^2
    lens_pot = results.get_lens_potential_cls(lmax)
    cl_pp = lens_pot[:,0]  # phi-phi, dimensionless
    l_arr = np.arange(lmax+1)
    # Pad cl_bb and cl_pp to lmax+1 if needed
    if cl_bb.shape[0] < lmax+1:
        cl_bb = np.pad(cl_bb, (0, lmax+1-cl_bb.shape[0]), 'constant')
    if cl_pp.shape[0] < lmax+1:
        cl_pp = np.pad(cl_pp, (0, lmax+1-cl_pp.shape[0]), 'constant')
    return l_arr, cl_bb, cl_pp

def compute_residual_phi(cl_pp, n0):
    r"""
    Compute the residual lensing potential power spectrum.

    Parameters
    ----------
    cl_pp : ndarray
        Lensing potential power spectrum (dimensionless), length lmax+1.
    n0 : ndarray
        Lensing noise power spectrum (dimensionless), length lmax+1.

    Returns
    -------
    cl_pp_res : ndarray
        Residual lensing potential power spectrum, length lmax+1.
    """
    # Avoid division by zero
    denom = cl_pp + n0
    mask = denom > 0
    cl_pp_res = np.zeros_like(cl_pp)
    cl_pp_res[mask] = cl_pp[mask] * (1.0 - (cl_pp[mask] / denom[mask]))
    return cl_pp_res

def camb_get_delensed_bb(cosmo_params, lmax, cl_pp_res):
    r"""
    Run CAMB to get the delensed BB power spectrum using the residual lensing potential.

    Parameters
    ----------
    cosmo_params : dict
        Dictionary of cosmological parameters.
    lmax : int
        Maximum multipole.
    cl_pp_res : ndarray
        Residual lensing potential power spectrum, length lmax+1.

    Returns
    -------
    cl_bb_delensed : ndarray
        Delensed BB power spectrum (muK^2), length lmax+1.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=cosmo_params['H0'],
                       ombh2=cosmo_params['ombh2'],
                       omch2=cosmo_params['omch2'],
                       mnu=cosmo_params['mnu'],
                       omk=cosmo_params['omk'],
                       tau=cosmo_params['tau'])
    pars.InitPower.set_params(As=cosmo_params['As'], ns=cosmo_params['ns'])
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    # Set custom lensing potential
    pars.WantTensors = False
    pars.Want_CMB_lensing = True
    pars.set_custom_lens_potential(cl_pp_res)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    lensed = powers['total']
    cl_bb_delensed = lensed[:,2]  # muK^2
    # Pad if needed
    if cl_bb_delensed.shape[0] < lmax+1:
        cl_bb_delensed = np.pad(cl_bb_delensed, (0, lmax+1-cl_bb_delensed.shape[0]), 'constant')
    return cl_bb_delensed

def main():
    r"""
    Main function to compute the delensing efficiency of the CMB B-mode power spectrum.
    Saves the results in 'data/result.csv' and prints a summary.
    """
    # Cosmological parameters (units in comments)
    cosmo_params = {
        'H0': 67.5,         # Hubble constant [km/s/Mpc]
        'ombh2': 0.022,     # Baryon density [dimensionless]
        'omch2': 0.122,     # Cold dark matter density [dimensionless]
        'mnu': 0.06,        # Neutrino mass sum [eV]
        'omk': 0.0,         # Curvature [dimensionless]
        'tau': 0.06,        # Optical depth [dimensionless]
        'As': 2e-9,         # Scalar amplitude [dimensionless]
        'ns': 0.965         # Scalar spectral index [dimensionless]
    }
    lmax = 2000

    # Load N0
    n0_filepath = '/Users/antoidicherianlonappan/Software/Benchmarks/data/N0.csv'
    l_arr_n0, n0 = load_n0_csv(n0_filepath, lmax)

    # Get lensed BB and phi-phi from CAMB
    l_arr, cl_bb, cl_pp = camb_get_cls_and_phi(cosmo_params, lmax)

    # Ensure arrays are same length
    if n0.shape[0] != cl_pp.shape[0]:
        minlen = min(n0.shape[0], cl_pp.shape[0])
        n0 = n0[:minlen]
        cl_pp = cl_pp[:minlen]
        cl_bb = cl_bb[:minlen]
        l_arr = l_arr[:minlen]

    # Compute residual lensing potential
    cl_pp_res = compute_residual_phi(cl_pp, n0)

    # Pad cl_pp_res to lmax+1 if needed
    if cl_pp_res.shape[0] < lmax+1:
        cl_pp_res = np.pad(cl_pp_res, (0, lmax+1-cl_pp_res.shape[0]), 'constant')

    # Get delensed BB
    cl_bb_delensed = camb_get_delensed_bb(cosmo_params, lmax, cl_pp_res)

    # Compute delensing efficiency for l=2 to l=100
    lmin_eff = 2
    lmax_eff = 100
    l_eff = np.arange(lmin_eff, lmax_eff+1)
    cl_bb_eff = cl_bb[lmin_eff:lmax_eff+1]
    cl_bb_delensed_eff = cl_bb_delensed[lmin_eff:lmax_eff+1]
    # Avoid division by zero
    mask = cl_bb_eff > 0
    delensing_efficiency = np.zeros_like(cl_bb_eff)
    delensing_efficiency[mask] = 100.0 * (cl_bb_eff[mask] - cl_bb_delensed_eff[mask]) / cl_bb_eff[mask]

    # Save results
    result_df = pd.DataFrame({'l': l_eff, 'delensing_efficiency': delensing_efficiency})
    result_df.to_csv('data/result.csv', index=False)

    # Print summary
    np.set_printoptions(precision=4, suppress=True)
    print("Delensing efficiency (percent) for l=2 to l=100:")
    print(result_df)

if __name__ == "__main__":
    main()