# filename: codebase/delensing_efficiency.py
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

def load_n0_csv(filename, lmax):
    r"""
    Load the lensing noise power spectrum N0 from a CSV file and convert to C_ell^{N0}.

    Parameters
    ----------
    filename : str
        Path to the CSV file containing columns 'l' and 'Nl'.
    lmax : int
        Maximum multipole to consider.

    Returns
    -------
    ell : ndarray
        Multipole moments (int) from 2 to lmax.
    n0_cl : ndarray
        Lensing noise power spectrum C_ell^{N0} (dimensionless), length lmax+1.
    """
    df = pd.read_csv(filename)
    ell = df['l'].values.astype(int)
    Nl = df['Nl'].values
    # Nl = N0 * (ell*(ell+1))^2 / (2pi) => N0 = Nl * (2pi) / (ell*(ell+1))^2
    N0 = np.zeros(lmax+1)
    valid = (ell >= 2) & (ell <= lmax)
    ell_valid = ell[valid]
    Nl_valid = Nl[valid]
    N0[ell_valid] = Nl_valid * (2.0 * np.pi) / (ell_valid * (ell_valid + 1))**2
    return np.arange(lmax+1), N0

def camb_get_cls(params, lmax, clpp_override=None):
    r"""
    Run CAMB to get lensed and delensed CMB power spectra.

    Parameters
    ----------
    params : camb.CAMBparams
        CAMB parameters object.
    lmax : int
        Maximum multipole.
    clpp_override : ndarray or None
        If provided, use this as the lensing potential power spectrum (C_ell^{\phi\phi}) for delensing.

    Returns
    -------
    ell : ndarray
        Multipole moments (int) from 2 to lmax.
    cl_bb_lensed : ndarray
        Lensed B-mode power spectrum (muK^2), length lmax+1.
    cl_pp : ndarray
        Lensing potential power spectrum (dimensionless), length lmax+1.
    cl_bb_delensed : ndarray or None
        Delensed B-mode power spectrum (muK^2), length lmax+1, or None if clpp_override is None.
    """
    # Set up lensing
    params.set_for_lmax(lmax, lens_potential_accuracy=1)
    results = camb.get_results(params)
    powers = results.get_cmb_power_spectra(params, CMB_unit='muK', lmax=lmax, raw_cl=True)
    cl_bb_lensed = np.zeros(lmax+1)
    cl_pp = np.zeros(lmax+1)
    # Lensed BB
    cl_bb_lensed[2:] = powers['total'][2:, 2]  # 0: TT, 1: EE, 2: BB, 3: TE
    # Lensing potential
    cl_pp[2:] = results.get_lens_potential_cls(lmax=lmax, raw_cl=True)[2:, 0]  # 0: phi-phi

    cl_bb_delensed = None
    if clpp_override is not None:
        # Set the lensing potential to the residual and recompute
        # This requires a new CAMB run with the modified clpp
        # Use the transfer_class to set custom lensing potential
        # Note: This is an advanced feature; we use the 'CustomLensPotentialCls' option
        params2 = params.copy()
        params2.WantTensors = False
        params2.Want_CMB_lensing = True
        params2.set_for_lmax(lmax, lens_potential_accuracy=1)
        params2.CustomLensPotentialCls = clpp_override
        results2 = camb.get_results(params2)
        powers2 = results2.get_cmb_power_spectra(params2, CMB_unit='muK', lmax=lmax, raw_cl=True)
        cl_bb_delensed = np.zeros(lmax+1)
        cl_bb_delensed[2:] = powers2['total'][2:, 2]
    return np.arange(lmax+1), cl_bb_lensed, cl_pp, cl_bb_delensed

def main():
    r"""
    Main function to compute the delensing efficiency of the CMB B-mode polarization power spectrum.
    """
    # Cosmological parameters
    H0 = 67.5  # km/s/Mpc
    ombh2 = 0.022
    omch2 = 0.122
    mnu = 0.06  # eV
    omk = 0.0
    tau = 0.06
    As = 2e-9
    ns = 0.965
    lmax = 2000

    # Load N0
    n0_file = '/Users/antoidicherianlonappan/Software/Benchmarks/data/N0.csv'
    ell, n0_cl = load_n0_csv(n0_file, lmax)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.WantTensors = False
    pars.Want_CMB_lensing = True
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)

    # Get lensed BB and lensing potential
    ell, cl_bb_lensed, cl_pp, _ = camb_get_cls(pars, lmax)

    # Convert cl_pp to (l(l+1))^2/(2pi) units for plotting/consistency
    cl_pp_plot = np.zeros_like(cl_pp)
    l = np.arange(lmax+1)
    valid = l >= 2
    cl_pp_plot[valid] = cl_pp[valid] * (l[valid]*(l[valid]+1))**2 / (2.0*np.pi)

    # Ensure n0_cl and cl_pp are same length
    if n0_cl.shape[0] < lmax+1:
        n0_cl = np.pad(n0_cl, (0, lmax+1-n0_cl.shape[0]), 'constant')
    elif n0_cl.shape[0] > lmax+1:
        n0_cl = n0_cl[:lmax+1]

    # Residual lensing potential power spectrum
    cl_pp_res = np.zeros_like(cl_pp)
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = cl_pp + n0_cl
        mask = denom > 0
        cl_pp_res[mask] = cl_pp[mask] * (1.0 - (cl_pp[mask] / denom[mask]))
        cl_pp_res[~mask] = 0.0

    # Pad cl_pp_res to length required by CAMB (lmax+1)
    if cl_pp_res.shape[0] < lmax+1:
        cl_pp_res = np.pad(cl_pp_res, (0, lmax+1-cl_pp_res.shape[0]), 'constant')
    elif cl_pp_res.shape[0] > lmax+1:
        cl_pp_res = cl_pp_res[:lmax+1]

    # Compute delensed BB using residual lensing potential
    _, _, _, cl_bb_delensed = camb_get_cls(pars, lmax, clpp_override=cl_pp_res)

    # Delensing efficiency
    lmin_eff = 2
    lmax_eff = 100
    eff_ells = np.arange(lmin_eff, lmax_eff+1)
    delensing_efficiency = np.zeros_like(eff_ells, dtype=float)
    for i, ell_val in enumerate(eff_ells):
        if cl_bb_lensed[ell_val] > 0:
            delensing_efficiency[i] = 100.0 * (cl_bb_lensed[ell_val] - cl_bb_delensed[ell_val]) / cl_bb_lensed[ell_val]
        else:
            delensing_efficiency[i] = 0.0

    # Save results
    result_df = pd.DataFrame({'l': eff_ells, 'delensing_efficiency': delensing_efficiency})
    result_df.to_csv('data/result.csv', index=False)

    # Print results
    np.set_printoptions(precision=6, suppress=True)
    pd.set_option('display.max_rows', None)
    print("\nDelensing efficiency (percent) for l=2 to l=100:")
    print(result_df)

    # Print summary of key spectra
    print("\nSummary of computed spectra:")
    print("lensed BB (muK^2) at l=2,10,50,100:", cl_bb_lensed[[2,10,50,100]])
    print("delensed BB (muK^2) at l=2,10,50,100:", cl_bb_delensed[[2,10,50,100]])
    print("lensing potential (l(l+1))^2/(2pi) at l=2,10,50,100:", cl_pp_plot[[2,10,50,100]])
    print("residual lensing potential (l(l+1))^2/(2pi) at l=2,10,50,100:",
          cl_pp_res[[2,10,50,100]] * (np.array([2,10,50,100]) * (np.array([2,10,50,100]) + 1))**2 / (2.0 * np.pi))


if __name__ == "__main__":
    main()