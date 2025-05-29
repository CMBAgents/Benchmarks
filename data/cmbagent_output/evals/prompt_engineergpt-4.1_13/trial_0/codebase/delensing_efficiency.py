# filename: codebase/delensing_efficiency.py
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
    ell : ndarray
        Array of multipole moments (int) up to lmax.
    n0_cl : ndarray
        Array of C_ell^{N0} (dimensionless, same units as C_ell^{\phi\phi}).
    """
    df = pd.read_csv(filepath)
    ell = df['l'].values.astype(int)
    Nl = df['Nl'].values
    mask = ell <= lmax
    ell = ell[mask]
    Nl = Nl[mask]
    # Nl = N0 * (ell*(ell+1))^2/(2pi) => N0 = Nl * (2pi)/(ell*(ell+1))^2
    n0_cl = np.zeros(lmax+1)
    n0_cl[ell] = Nl * (2.0 * np.pi) / (ell * (ell + 1))**2
    return ell, n0_cl

def get_camb_results(lmax, params_dict):
    r"""
    Run CAMB to get lensed B-mode and lensing potential power spectra.

    Parameters
    ----------
    lmax : int
        Maximum multipole.
    params_dict : dict
        Dictionary of cosmological parameters.

    Returns
    -------
    ell : ndarray
        Multipole array (int) from 0 to lmax.
    cl_bb_lensed : ndarray
        Lensed B-mode power spectrum (muK^2).
    cl_pp : ndarray
        Lensing potential power spectrum (dimensionless).
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=params_dict['H0'],
                       ombh2=params_dict['ombh2'],
                       omch2=params_dict['omch2'],
                       mnu=params_dict['mnu'],
                       omk=params_dict['omk'],
                       tau=params_dict['tau'])
    pars.InitPower.set_params(As=params_dict['As'], ns=params_dict['ns'])
    pars.set_for_lmax(lmax, lens_potential_accuracy=2)
    pars.WantTensors = False
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    cl = powers['total']
    # cl shape: (lmax+1, 4): TT, EE, BB, TE
    cl_bb_lensed = cl[:,2]  # muK^2
    # Lensing potential
    cl_pp = results.get_lens_potential_cls(lmax=lmax, raw_cl=True)[:,0]  # dimensionless
    ell = np.arange(cl_bb_lensed.size)
    return ell, cl_bb_lensed, cl_pp

def compute_residual_phi(cl_pp, n0_cl):
    r"""
    Compute the residual lensing potential power spectrum.

    Parameters
    ----------
    cl_pp : ndarray
        Lensing potential power spectrum (dimensionless).
    n0_cl : ndarray
        Lensing noise power spectrum (dimensionless).

    Returns
    -------
    cl_pp_res : ndarray
        Residual lensing potential power spectrum (dimensionless).
    """
    # Avoid division by zero
    denom = cl_pp + n0_cl
    mask = denom > 0
    cl_pp_res = np.zeros_like(cl_pp)
    cl_pp_res[mask] = cl_pp[mask] * (1.0 - (cl_pp[mask] / denom[mask]))
    return cl_pp_res

def pad_to_length(arr, length):
    r"""
    Pad or truncate an array to a given length.

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

def get_delensed_bb(lmax, params_dict, cl_pp_res):
    r"""
    Compute the delensed B-mode power spectrum using the residual lensing potential.

    Parameters
    ----------
    lmax : int
        Maximum multipole.
    params_dict : dict
        Cosmological parameters.
    cl_pp_res : ndarray
        Residual lensing potential power spectrum (dimensionless).

    Returns
    -------
    cl_bb_delensed : ndarray
        Delensed B-mode power spectrum (muK^2).
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=params_dict['H0'],
                       ombh2=params_dict['ombh2'],
                       omch2=params_dict['omch2'],
                       mnu=params_dict['mnu'],
                       omk=params_dict['omk'],
                       tau=params_dict['tau'])
    pars.InitPower.set_params(As=params_dict['As'], ns=params_dict['ns'])
    pars.set_for_lmax(lmax, lens_potential_accuracy=2)
    pars.WantTensors = False
    # Set custom lensing potential
    pars.set_custom_lens_potential(cl_pp_res)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    cl = powers['total']
    cl_bb_delensed = cl[:,2]  # muK^2
    return cl_bb_delensed

def main():
    r"""
    Main function to compute and save the delensing efficiency for CMB B-modes.
    """
    # Cosmological parameters
    params_dict = {
        'H0': 67.5,         # km/s/Mpc
        'ombh2': 0.022,     # Omega_b h^2
        'omch2': 0.122,     # Omega_c h^2
        'mnu': 0.06,        # eV
        'omk': 0.0,         # curvature
        'tau': 0.06,        # optical depth
        'As': 2e-9,         # scalar amplitude
        'ns': 0.965         # scalar spectral index
    }
    lmax = 2000

    # 1. Load N0
    n0_filepath = '/Users/antoidicherianlonappan/Software/Benchmarks/data/N0.csv'
    ell_n0, n0_cl = load_n0_csv(n0_filepath, lmax)

    # 2. Get lensed BB and lensing potential
    ell, cl_bb_lensed, cl_pp = get_camb_results(lmax, params_dict)

    # 3. Ensure cl_pp and n0_cl have same length (lmax+1)
    cl_pp = pad_to_length(cl_pp, lmax+1)
    n0_cl = pad_to_length(n0_cl, lmax+1)

    # 4. Compute residual lensing potential
    cl_pp_res = compute_residual_phi(cl_pp, n0_cl)

    # 5. Pad residual lensing potential to CAMB's required length (lmax+1)
    cl_pp_res = pad_to_length(cl_pp_res, lmax+1)

    # 6. Compute delensed BB
    cl_bb_delensed = get_delensed_bb(lmax, params_dict, cl_pp_res)

    # 7. Compute delensing efficiency for l=2 to 100
    lmin_eff = 2
    lmax_eff = 100
    ell_eff = np.arange(lmin_eff, lmax_eff+1)
    cl_bb_lensed_eff = cl_bb_lensed[lmin_eff:lmax_eff+1]
    cl_bb_delensed_eff = cl_bb_delensed[lmin_eff:lmax_eff+1]
    # Avoid division by zero
    mask = cl_bb_lensed_eff != 0
    delensing_efficiency = np.zeros_like(cl_bb_lensed_eff)
    delensing_efficiency[mask] = 100.0 * (cl_bb_lensed_eff[mask] - cl_bb_delensed_eff[mask]) / cl_bb_lensed_eff[mask]

    # Save results
    result_df = pd.DataFrame({
        'l': ell_eff,
        'delensing_efficiency': delensing_efficiency
    })
    result_path = 'data/result.csv'
    result_df.to_csv(result_path, index=False)

    # Print results
    np.set_printoptions(precision=6, suppress=True)
    pd.set_option('display.max_rows', None)
    print("\nDelensing efficiency (percent) for l=2 to l=100:")
    print(result_df)
    print("\nSaved delensing efficiency results to " + result_path)


if __name__ == "__main__":
    main()