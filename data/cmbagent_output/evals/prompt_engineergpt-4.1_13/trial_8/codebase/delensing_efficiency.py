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
    mask = l <= lmax
    l = l[mask]
    Nl = Nl[mask]
    # Convert Nl to N0: Nl = N0 * (l(l+1))^2 / (2pi) => N0 = Nl * (2pi) / (l(l+1))^2
    factor = (2.0 * np.pi) / (l * (l + 1))**2
    n0 = Nl * factor
    return l, n0

def get_camb_cls(params, lmax):
    r"""
    Run CAMB to get lensed and unlensed CMB power spectra and lensing potential.

    Parameters
    ----------
    params : dict
        Cosmological parameters.
    lmax : int
        Maximum multipole.

    Returns
    -------
    l_arr : ndarray
        Multipole array (int) from 0 to lmax.
    cl_bb_lensed : ndarray
        Lensed B-mode power spectrum (float), units: microK^2.
    cl_pp : ndarray
        Lensing potential power spectrum (float), units: dimensionless.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=params['H0'], ombh2=params['ombh2'], omch2=params['omch2'],
                       mnu=params['mnu'], omk=params['omk'], tau=params['tau'])
    pars.InitPower.set_params(As=params['As'], ns=params['ns'])
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.WantTensors = False
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    l_arr = np.arange(powers['lensed_scalar'].shape[0])
    cl_bb_lensed = powers['lensed_scalar'][:, 2]  # BB
    # Get lensing potential power spectrum
    cl_pp = results.get_lens_potential_cls(lmax=lmax)[:, 0]  # phi-phi
    return l_arr, cl_bb_lensed, cl_pp

def pad_to_length(arr, length):
    r"""
    Pad an array with zeros to a specified length.

    Parameters
    ----------
    arr : ndarray
        Input array.
    length : int
        Desired length.

    Returns
    -------
    arr_padded : ndarray
        Array padded to the desired length.
    """
    arr_padded = np.zeros(length)
    arr_padded[:min(len(arr), length)] = arr[:min(len(arr), length)]
    return arr_padded

def compute_delensed_bb(lmax, params, cl_pp_res):
    r"""
    Compute the delensed B-mode power spectrum using the residual lensing potential.

    Parameters
    ----------
    lmax : int
        Maximum multipole.
    params : dict
        Cosmological parameters.
    cl_pp_res : ndarray
        Residual lensing potential power spectrum.

    Returns
    -------
    cl_bb_delensed : ndarray
        Delensed B-mode power spectrum (float), units: microK^2.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=params['H0'], ombh2=params['ombh2'], omch2=params['omch2'],
                       mnu=params['mnu'], omk=params['omk'], tau=params['tau'])
    pars.InitPower.set_params(As=params['As'], ns=params['ns'])
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
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
    """
    # Cosmological parameters
    params = {
        'H0': 67.5,         # Hubble constant [km/s/Mpc]
        'ombh2': 0.022,     # Omega_b h^2
        'omch2': 0.122,     # Omega_c h^2
        'mnu': 0.06,        # sum m_nu [eV]
        'omk': 0.0,         # Omega_k
        'tau': 0.06,        # Optical depth
        'As': 2e-9,         # Scalar amplitude
        'ns': 0.965         # Scalar spectral index
    }
    lmax = 2000

    # 1. Load N0
    n0_filepath = '/Users/antoidicherianlonappan/Software/Benchmarks/data/N0.csv'
    l_n0, n0 = load_n0_csv(n0_filepath, lmax)
    # Pad n0 to lmax+1 (CAMB arrays are 0..lmax)
    n0_full = np.zeros(lmax + 1)
    n0_full[l_n0] = n0

    # 2. Get lensed BB and lensing potential
    l_arr, cl_bb_lensed, cl_pp = get_camb_cls(params, lmax)

    # 3. Ensure cl_pp and n0_full have same length (lmax+1)
    cl_pp = pad_to_length(cl_pp, lmax + 1)
    n0_full = pad_to_length(n0_full, lmax + 1)

    # 4. Compute residual lensing potential
    # cl_pp_res = cl_pp * (1 - (cl_pp / (cl_pp + n0)))
    with np.errstate(divide='ignore', invalid='ignore'):
        cl_pp_res = cl_pp * (1.0 - (cl_pp / (cl_pp + n0_full)))
        cl_pp_res = np.nan_to_num(cl_pp_res, nan=0.0, posinf=0.0, neginf=0.0)

    # 5. Pad cl_pp_res to length required by CAMB (lmax+1)
    cl_pp_res = pad_to_length(cl_pp_res, lmax + 1)

    # 6. Compute delensed BB
    cl_bb_delensed = compute_delensed_bb(lmax, params, cl_pp_res)

    # 7. Compute delensing efficiency for l=2 to l=100
    l_out = np.arange(2, 101)
    cl_bb_lensed_out = cl_bb_lensed[l_out]
    cl_bb_delensed_out = cl_bb_delensed[l_out]
    with np.errstate(divide='ignore', invalid='ignore'):
        delensing_efficiency = 100.0 * (cl_bb_lensed_out - cl_bb_delensed_out) / cl_bb_lensed_out
        delensing_efficiency = np.nan_to_num(delensing_efficiency, nan=0.0, posinf=0.0, neginf=0.0)

    # Save results
    result_df = pd.DataFrame({'l': l_out, 'delensing_efficiency': delensing_efficiency})
    result_df.to_csv('data/result.csv', index=False)

    # Print results
    print("Delensing efficiency (percent) for l=2 to l=100:")
    print(result_df.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()