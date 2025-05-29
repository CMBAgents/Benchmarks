# filename: codebase/cmb_tt_spectrum.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

# Ensure data directory exists
database_path = "data"
if not os.path.exists(database_path):
    os.makedirs(database_path)


def compute_cmb_tt_spectrum():
    r"""
    Computes the CMB temperature power spectrum (C_l^{TT}) for the specified cosmology
    using CAMB, and saves the result as a CSV file.

    Returns
    -------
    None
    """
    # Cosmological parameters
    H0 = 70.0  # Hubble constant [km/s/Mpc]
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
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    # powers['total'] is (lmax+1, 4): columns are TT, EE, BB, TE

    # Extract TT spectrum
    cl_tt = powers['total'][:, 0]  # TT spectrum in muK^2

    # l values: CAMB returns from l=0 to l=lmax
    ell = np.arange(cl_tt.shape[0])

    # Select l=2 to l=3000
    mask = (ell >= lmin) & (ell <= lmax)
    ell = ell[mask]
    cl_tt = cl_tt[mask]

    # Save to CSV
    df = pd.DataFrame({'l': ell, 'TT': cl_tt})
    output_file = os.path.join(database_path, "result.csv")
    df.to_csv(output_file, index=False)

    # Print summary
    pd.set_option("display.precision", 8)
    pd.set_option("display.max_rows", 10)
    print("CMB TT power spectrum (C_l^{TT}) in microkelvin^2, l=2 to l=3000:")
    print(df.head(5))
    print("...")
    print(df.tail(5))
    print("\nSaved full result to " + output_file)


if __name__ == "__main__":
    compute_cmb_tt_spectrum()