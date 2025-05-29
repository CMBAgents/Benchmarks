# filename: codebase/cmb_ee_spectrum.py
import os
import numpy as np
import pandas as pd

import camb
from camb import model, initialpower

def compute_cmb_ee_spectrum():
    """
    Compute the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for the specified cosmology
    and save the results in 'data/result.csv'.

    Returns
    -------
    None
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density Omega_b h^2
    omch2 = 0.122  # Cold dark matter density Omega_c h^2
    tau = 0.1  # Optical depth to reionization (dimensionless)
    ns = 0.95  # Scalar spectral index (dimensionless)
    # Scalar amplitude: 1.8e-9 * exp(2 * tau)
    As = 1.8e-9 * np.exp(2.0 * tau)  # Scalar amplitude (dimensionless)

    # Multipole range
    lmin = 2
    lmax = 100

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)

    # Set reionization model: Exponential with exponent power 2
    pars.Reion.use_optical_depth = True
    pars.Reion.reionization = True
    pars.Reion.reionization_zstart = 15.0  # Default, not used for exp model
    pars.Reion.reionization_exponent = 2.0  # Exponent power
    pars.Reion.reionization_model = camb.reionization.ReionizationModel.Exp
    pars.Reion.optical_depth = tau

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax)
    EE = powers['total'][:, 1]  # EE is column 1

    # l values: CAMB returns from l=0, so slice from l=2 to l=100
    l_vals = np.arange(2, lmax + 1)
    EE_vals = EE[2:lmax + 1]  # EE[2] is l=2

    # Compute l(l+1)C_l^{EE}/(2pi) [uK^2]
    factor = l_vals * (l_vals + 1) / (2.0 * np.pi)
    EE_power = factor * EE_vals  # [uK^2]

    # Prepare DataFrame
    df = pd.DataFrame({'l': l_vals, 'EE': EE_power})

    # Ensure data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save to CSV
    csv_path = os.path.join(data_dir, "result.csv")
    df.to_csv(csv_path, index=False)

    # Print results to console
    np.set_printoptions(precision=6, suppress=True)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print("CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) [uK^2] for l=2 to l=100:")
    print(df)

if __name__ == "__main__":
    compute_cmb_ee_spectrum()
