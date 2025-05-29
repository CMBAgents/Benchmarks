# filename: codebase/delensing.py
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

def load_n0_csv(filepath, lmax):
    r"""
    Load the lensing noise power spectrum N0 from a CSV file and convert to C_ell^{N0}.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file containing columns 'l' and 'Nl'.
    lmax : int
        Maximum multipole to consider.
    
    Returns
    -------
    l_arr : ndarray
        Array of multipole moments (int).
    cl_n0 : ndarray
        Array of C_ell^{N0} values (float), length lmax+1, units: dimensionless.
    """
    df = pd.read_csv(filepath)
    l = df['l'].values.astype(int)
    Nl = df['Nl'].values
    # Nl = N0 * (l(l+1))^2 / (2pi) => N0 = Nl * (2pi) / (l(l+1))^2
    cl_n0 = np.zeros(lmax+1)
    valid = (l <= lmax)
    l = l[valid]
    Nl = Nl[valid]
    factor = (2.0 * np.pi) / ((l * (l + 1)) ** 2)
    cl_n0[l] = Nl * factor
    return l, cl_n0

def get_camb_results(params, lmax, clpp_override=None):
    r"""
    Run CAMB to get lensed BB and lensing potential power spectra.
    
    Parameters
    ----------
    params : dict
        Cosmological parameters.
    lmax : int
        Maximum multipole.
    clpp_override : ndarray or None
        If provided, use this as the lensing potential power spectrum (for delensed case).
    
    Returns
    -------
    l : ndarray
        Multipole moments (int), from 0 to lmax.
    cl_bb : ndarray
        Lensed BB power spectrum (float), units: microK^2.
    cl_pp : ndarray
        Lensing potential power spectrum (float), dimensionless.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=params['H0'], ombh2=params['ombh2'], omch2=params['omch2'],
                       mnu=params['mnu'], omk=params['omk'], tau=params['tau'])
    pars.InitPower.set_params(As=params['As'], ns=params['ns'])
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.WantTensors = False
    pars.Want_CMB_lensing = True
    if clpp_override is not None:
        # Use custom lensing potential
        pars.set_custom_lens_potential(clpp_override)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    cl_bb = powers['lensed_scalar'][:, 2]  # BB
    cl_pp = results.get_lens_potential_cls(lmax=lmax)[:, 0]  # phi-phi
    l = np.arange(cl_bb.size)
    return l, cl_bb, cl_pp

def compute_residual_clpp(cl_pp, cl_n0):
    r"""
    Compute the residual lensing potential power spectrum.
    
    Parameters
    ----------
    cl_pp : ndarray
        Lensing potential power spectrum (dimensionless).
    cl_n0 : ndarray
        Lensing noise power spectrum (dimensionless).
    
    Returns
    -------
    cl_pp_res : ndarray
        Residual lensing potential power spectrum (dimensionless).
    """
    # Avoid division by zero
    denom = cl_pp + cl_n0
    mask = denom > 0
    cl_pp_res = np.zeros_like(cl_pp)
    cl_pp_res[mask] = cl_pp[mask] * (1.0 - (cl_pp[mask] / denom[mask]))
    return cl_pp_res

def main():
    r"""
    Main function to compute delensing efficiency and save results.
    """
    # Cosmological parameters
    params = {
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
    # 1. Load N0 and convert to C_ell^{N0}
    l_n0, cl_n0 = load_n0_csv(n0_filepath, lmax)
    # 2. Get lensed BB and phi-phi from CAMB
    l, cl_bb_lensed, cl_pp = get_camb_results(params, lmax)
    # 3. Ensure cl_pp and cl_n0 have same length (lmax+1)
    cl_pp = cl_pp[:lmax+1]
    cl_n0 = cl_n0[:lmax+1]
    # 4. Compute residual lensing potential
    cl_pp_res = compute_residual_clpp(cl_pp, cl_n0)
    # 5. Pad residual lensing potential to match CAMB's expected length (lmax+1)
    cl_pp_res_padded = np.zeros(lmax+1)
    cl_pp_res_padded[:cl_pp_res.size] = cl_pp_res
    # 6. Compute delensed BB using CAMB with residual lensing potential
    l, cl_bb_delensed, _ = get_camb_results(params, lmax, clpp_override=cl_pp_res_padded)
    # 7. Compute delensing efficiency for l=2 to l=100
    l_range = np.arange(2, 101)
    cl_bb_lensed_sel = cl_bb_lensed[l_range]
    cl_bb_delensed_sel = cl_bb_delensed[l_range]
    delensing_efficiency = 100.0 * (cl_bb_lensed_sel - cl_bb_delensed_sel) / cl_bb_lensed_sel
    # Save results
    result_df = pd.DataFrame({'l': l_range, 'delensing_efficiency': delensing_efficiency})
    result_df.to_csv('data/result.csv', index=False)
    # Print results
    np.set_printoptions(precision=6, suppress=True)
    pd.set_option('display.max_rows', 100)
    print("\nDelensing efficiency (percent) for l=2 to l=100:")
    print(result_df)
    print("\nSaved delensing efficiency results to data/result.csv")

if __name__ == "__main__":
    main()