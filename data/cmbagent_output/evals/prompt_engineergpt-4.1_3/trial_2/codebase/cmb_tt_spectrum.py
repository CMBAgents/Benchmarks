# filename: codebase/cmb_tt_spectrum.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

def compute_cmb_tt_spectrum():
    r"""
    Compute the CMB temperature power spectrum (C_l^{TT}) for a flat Lambda CDM cosmology
    using CAMB with the specified cosmological parameters.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns:
            - 'l': Multipole moment (integer, 2 to 3000)
            - 'TT': Raw temperature power spectrum in microkelvin^2 (μK^2)
    """
    # Cosmological parameters
    H0 = 74.0  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density parameter [dimensionless]
    omch2 = 0.122  # Cold dark matter density parameter [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    As = 2e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]

    # Multipole range
    lmin = 2
    lmax = 3000

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.WantTensors = False

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
    tt = powers['unlensed_scalar'][lmin:lmax+1, 0]  # TT spectrum, μK^2

    # Prepare DataFrame
    l_vals = np.arange(lmin, lmax+1)
    df = pd.DataFrame({'l': l_vals, 'TT': tt})

    return df


def save_results_to_csv(df, filename):
    r"""
    Save the DataFrame to a CSV file in the data/ directory.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns 'l' and 'TT'.
    filename : str
        Name of the CSV file to save (should include .csv extension).
    """
    outdir = "data"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    filepath = os.path.join(outdir, filename)
    df.to_csv(filepath, index=False)
    print("CMB TT power spectrum saved to " + filepath)
    print("First 5 rows:\n" + str(df.head()))
    print("Last 5 rows:\n" + str(df.tail()))


if __name__ == "__main__":
    df = compute_cmb_tt_spectrum()
    save_results_to_csv(df, "result.csv")
