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
        Array of N0 values (float) up to lmax, in units of C_l^{\phi\phi}.
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
    # Pad to lmax+1 (CAMB arrays are 0..lmax)
    n0_arr = np.zeros(lmax + 1)
    n0_arr[l] = N0
    l_arr = np.arange(lmax + 1)
    return l_arr, n0_arr

def camb_get_cls(params, lmax):
    r"""
    Run CAMB to get lensed B-mode and lensing potential power spectra.

    Parameters
    ----------
    params : dict
        Cosmological parameters.
    lmax : int
        Maximum multipole.

    Returns
    -------
    l : ndarray
        Multipole array (int) from 0 to lmax.
    cl_bb : ndarray
        Lensed B-mode power spectrum (muK^2).
    cl_pp : ndarray
        Lensing potential power spectrum (C_l^{\phi\phi}).
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=params['H0'], ombh2=params['ombh2'], omch2=params['omch2'],
                       mnu=params['mnu'], omk=params['omk'], tau=params['tau'])
    pars.InitPower.set_params(As=params['As'], ns=params['ns'])
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.WantTensors = False
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    lens_pot_cl = results.get_lens_potential_cls(lmax=lmax)
    # lensed BB
    cl_bb = powers['total'][:, 2]  # BB is index 2
    # lensing potential
    cl_pp = lens_pot_cl[:, 0]  # phi-phi
    l = np.arange(cl_bb.size)
    return l, cl_bb, cl_pp

def camb_delensed_bb(params, lmax, cl_pp_res):
    r"""
    Run CAMB to get delensed B-mode power spectrum using a modified lensing potential.

    Parameters
    ----------
    params : dict
        Cosmological parameters.
    lmax : int
        Maximum multipole.
    cl_pp_res : ndarray
        Residual lensing potential power spectrum (C_l^{\phi\phi}), length lmax+1.

    Returns
    -------
    l : ndarray
        Multipole array (int) from 0 to lmax.
    cl_bb_delensed : ndarray
        Delensed B-mode power spectrum (muK^2).
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
    cl_bb_delensed = powers['total'][:, 2]  # BB is index 2
    l = np.arange(cl_bb_delensed.size)
    return l, cl_bb_delensed

def main():
    r"""
    Main function to compute the delensing efficiency of the CMB B-mode polarization power spectrum.
    Saves the results in 'data/result.csv' and prints detailed results to the console.
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

    # 1. Load N0
    n0_filepath = '/Users/antoidicherianlonappan/Software/Benchmarks/data/N0.csv'
    l_arr, n0_arr = load_n0_csv(n0_filepath, lmax)

    # 2. Get lensed BB and lensing potential
    l, cl_bb, cl_pp = camb_get_cls(params, lmax)

    # 3. Ensure cl_pp and n0_arr have same length (lmax+1)
    if cl_pp.shape[0] < lmax + 1:
        cl_pp = np.pad(cl_pp, (0, lmax + 1 - cl_pp.shape[0]), mode='constant')
    if n0_arr.shape[0] < lmax + 1:
        n0_arr = np.pad(n0_arr, (0, lmax + 1 - n0_arr.shape[0]), mode='constant')

    # 4. Residual lensing potential
    cl_pp_res = cl_pp * (1.0 - (cl_pp / (cl_pp + n0_arr)))
    # 5. Pad to lmax+1 (already done above)

    # 6. Delensed BB
    l_del, cl_bb_delensed = camb_delensed_bb(params, lmax, cl_pp_res)

    # 7. Delensing efficiency for l=2 to l=100
    lmin_eff = 2
    lmax_eff = 100
    mask_eff = (l >= lmin_eff) & (l <= lmax_eff)
    delensing_efficiency = 100.0 * (cl_bb[mask_eff] - cl_bb_delensed[mask_eff]) / cl_bb[mask_eff]

    # Save results
    result_df = pd.DataFrame({
        'l': l[mask_eff],
        'delensing_efficiency': delensing_efficiency
    })
    result_df.to_csv('data/result.csv', index=False)

    # Print results
    np.set_printoptions(precision=6, suppress=True)
    pd.set_option('display.float_format', lambda x: '%.6f' % x)
    print("\nDelensing efficiency results (l=2 to l=100):")
    print(result_df)
    print("\nSaved delensing efficiency to data/result.csv")
    print("\nSummary:")
    print("Lensed BB (muK^2) at l=2: " + str(cl_bb[2]))
    print("Delensed BB (muK^2) at l=2: " + str(cl_bb_delensed[2]))
    print("Lensed BB (muK^2) at l=100: " + str(cl_bb[100]))
    print("Delensed BB (muK^2) at l=100: " + str(cl_bb_delensed[100]))
    print("Delensing efficiency at l=2: " + str(delensing_efficiency[0]) + " %")
    print("Delensing efficiency at l=100: " + str(delensing_efficiency[-1]) + " %")

if __name__ == "__main__":
    main()