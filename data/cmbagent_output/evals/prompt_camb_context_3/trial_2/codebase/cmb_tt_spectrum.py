# filename: codebase/cmb_tt_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_tt_spectrum(
    H0=74.0,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density parameter [dimensionless]
    omch2=0.122,            # Cold dark matter density parameter [dimensionless]
    mnu=0.06,               # Sum of neutrino masses [eV]
    omk=0.0,                # Curvature parameter [dimensionless]
    tau=0.06,               # Optical depth to reionization [dimensionless]
    As=2e-9,                # Scalar amplitude [dimensionless]
    ns=0.965,               # Scalar spectral index [dimensionless]
    lmax=3000,              # Maximum multipole moment [dimensionless]
    output_csv='data/result.csv'  # Output CSV file path
):
    r"""
    Compute the CMB raw temperature power spectrum (C_l^{TT}) for a flat Lambda CDM cosmology
    using CAMB, and save the results to a CSV file.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter (Omega_b h^2).
    omch2 : float
        Physical cold dark matter density parameter (Omega_c h^2).
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter (Omega_k).
    tau : float
        Optical depth to reionization.
    As : float
        Scalar amplitude of the primordial power spectrum.
    ns : float
        Scalar spectral index.
    lmax : int
        Maximum multipole moment to compute.
    output_csv : str
        Path to the output CSV file.

    Returns
    -------
    None
        The function saves the results to a CSV file and prints a summary to the console.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax=lmax)
    pars.DoLensing = True

    # Compute results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra (raw Cl, in muK^2)
    powers = results.get_cmb_power_spectra(
        params=pars,
        CMB_unit='muK',
        raw_cl=True,
        spectra=['lensed_scalar']
    )
    cl_tt = powers['lensed_scalar'][:, 0]  # TT is the first column

    # Prepare l and TT arrays for l=2..lmax
    l_vals = np.arange(2, lmax + 1)
    tt_vals = cl_tt[2:lmax + 1]

    # Save to CSV
    df = pd.DataFrame({'l': l_vals, 'TT': tt_vals})
    df.to_csv(output_csv, index=False)

    # Print summary to console
    print("CMB raw temperature power spectrum (C_l^{TT}) computed for flat Lambda CDM cosmology.")
    print("Cosmological parameters used:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  Sum m_nu = " + str(mnu) + " eV")
    print("  Omega_k = " + str(omk))
    print("  tau = " + str(tau))
    print("  As = " + str(As))
    print("  ns = " + str(ns))
    print("Multipole range: l = 2 to " + str(lmax))
    print("Results saved to: " + output_csv)
    print("First 5 rows of the result:")
    print(df.head())
    print("Last 5 rows of the result:")
    print(df.tail())


if __name__ == "__main__":
    compute_cmb_tt_spectrum()
