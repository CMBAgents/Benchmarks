# filename: codebase/cmb_tt_spectrum.py
import os
import numpy as np
import pandas as pd

# Ensure output directory exists
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

import camb
from camb import model, initialpower

def compute_cmb_tt_spectrum(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.02,             # Ω_b h^2
    omch2=0.122,            # Ω_c h^2
    mnu=0.06,               # Σ m_ν [eV]
    omk=0.0,                # Ω_k (flat)
    tau=0.06,               # Optical depth
    As=2e-9,                # Scalar amplitude
    ns=0.965,               # Scalar spectral index
    lmin=2,                 # Minimum multipole
    lmax=3000,              # Maximum multipole
    output_csv="data/result.csv"
):
    """
    Compute the CMB temperature power spectrum for given cosmological parameters.

    Parameters are as described above.

    Saves the result to output_csv.
    """
    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.WantTensors = False

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    totCL = powers['total']

    # l values: CAMB returns from l=0, so index accordingly
    ell = np.arange(totCL.shape[0])  # l=0 to lmax
    # Select l=2 to lmax
    l_range = np.arange(lmin, lmax+1)
    # Compute l(l+1)C_l^{TT}/(2π) in μK^2
    TT = totCL[lmin:lmax+1, 0]  # TT column, lmin to lmax
    TT_power = l_range * (l_range + 1) * TT / (2.0 * np.pi)  # μK^2

    # Save to CSV
    df = pd.DataFrame({'l': l_range, 'TT': TT_power})
    df.to_csv(output_csv, index=False)

    # Print summary
    np.set_printoptions(precision=6, suppress=True)
    pd.set_option("display.precision", 6)
    pd.set_option("display.max_rows", 10)
    print("CMB TT power spectrum (l(l+1)C_l^{TT}/(2π)) in μK^2, l=2 to l=3000:")
    print(df.head(5))
    print("...")
    print(df.tail(5))
    print("\nSaved result to " + output_csv)

if __name__ == "__main__":
    compute_cmb_tt_spectrum()