# filename: codebase/compute_cmb_tt_spectrum.py
import os
import numpy as np
import pandas as pd

# Ensure data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

try:
    import camb
    from camb import model, initialpower
except ImportError as e:
    print("CAMB package is required but not installed. Please install camb and rerun the code.")
    raise e


def compute_cmb_tt_spectrum():
    r"""
    Compute the CMB temperature power spectrum (C_l^{TT}) for a flat Lambda CDM cosmology
    using CAMB, with the following parameters:
        - Hubble constant (H0): 70 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Neutrino mass sum (mnu): 0.06 eV
        - Curvature (omk): 0
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965

    Computes the temperature power spectrum (C_l^{TT}) in units of microkelvin^2
    for multipole moments l=2 to l=3000, and saves the results in 'data/result.csv'.

    Returns
    -------
    None
    """
    # Cosmological parameters
    H0 = 70.0  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    As = 2e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(3000, lens_potential_accuracy=0)
    pars.WantTensors = False

    # Calculate results for these parameters
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000)
    totCL = powers['total']  # Shape: (lmax+1, 4) for TT, EE, BB, TE

    # l values start from 0, but CMB spectrum is meaningful for l >= 2
    l = np.arange(totCL.shape[0])  # l = 0 ... lmax
    lmin = 2
    lmax = 3000
    l_range = np.arange(lmin, lmax + 1)
    TT = totCL[lmin:lmax+1, 0]  # TT spectrum in muK^2

    # Save to CSV
    df = pd.DataFrame({'l': l_range, 'TT': TT})
    output_path = os.path.join("data", "result.csv")
    df.to_csv(output_path, index=False)

    # Print summary of results
    np.set_printoptions(precision=6, suppress=True)
    print("CMB TT power spectrum (C_l^{TT}) computed for l = 2 to 3000.")
    print("Results saved to: " + output_path)
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())


if __name__ == "__main__":
    compute_cmb_tt_spectrum()
