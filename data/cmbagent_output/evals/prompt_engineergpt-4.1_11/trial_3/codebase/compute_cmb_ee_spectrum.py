# filename: codebase/compute_cmb_ee_spectrum.py
import os
import numpy as np
import pandas as pd

import camb
from camb import model, initialpower

def compute_cmb_ee_spectrum():
    r"""
    Computes the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for a flat Lambda CDM cosmology
    using CAMB, with specified cosmological parameters and an exponential reionization model (exponent power 2).
    Saves the results in 'data/result.csv' with columns 'l' and 'EE' (in uK^2).
    """
    # Ensure data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density Omega_b h^2
    omch2 = 0.122  # Cold dark matter density Omega_c h^2
    tau = 0.1  # Optical depth to reionization (dimensionless)
    ns = 0.95  # Scalar spectral index (dimensionless)
    # Scalar amplitude A_s = 1.8e-9 * exp(2 * tau)
    As = 1.8e-9 * np.exp(2 * tau)  # dimensionless

    # Multipole range
    lmin = 2
    lmax = 100

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)

    # Set exponential reionization model with exponent power 2
    pars.Reion.use_optical_depth = True
    pars.Reion.reionization = True
    pars.Reion.reionization_scheme = camb.reionization.ReionizationScheme.Exp
    pars.Reion.exp_reion_exponent = 2.0

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total'])
    totCL = powers['total']

    # l values: CAMB returns Cls for l=0,...,lmax
    ells = np.arange(totCL.shape[0])  # l=0,...,lmax
    # EE spectrum is column 2 (index 2)
    EE = totCL[:,2]  # units: muK^2

    # Compute l(l+1)C_l^{EE}/(2pi) for l=2 to lmax
    lvals = np.arange(lmin, lmax+1)
    factor = lvals * (lvals + 1) / (2.0 * np.pi)
    EE_power = factor * EE[lmin:lmax+1]  # units: muK^2

    # Save to CSV
    df = pd.DataFrame({'l': lvals, 'EE': EE_power})
    output_csv = os.path.join(data_dir, "result.csv")
    df.to_csv(output_csv, index=False)

    # Print results to console
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", lambda x: "%.6e" % x)
    print("CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) [uK^2] for l=2 to l=100:")
    print(df)
    print("\nResults saved to " + output_csv)

if __name__ == "__main__":
    compute_cmb_ee_spectrum()
