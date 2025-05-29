# filename: codebase/camb_cmb_ee_spectrum.py
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower
import os

def compute_cmb_ee_spectrum():
    """
    Computes the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for a flat Lambda CDM cosmology
    using CAMB, with specified cosmological parameters and an exponential reionization model with exponent power 2.
    Saves the results to data/result.csv with columns:
    - l: Multipole moment (integer, 2 to 100)
    - EE: E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) in microkelvin^2 (uK^2)
    Also prints the results to the console.
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    tau = 0.1  # Optical depth to reionization [dimensionless]
    ns = 0.95  # Scalar spectral index [dimensionless]
    # Scalar amplitude As = 1.8e-9 * exp(2 * tau)
    As = 1.8e-9 * np.exp(2 * tau)  # [dimensionless]

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(100, lens_potential_accuracy=0)
    # Exponential reionization with exponent power 2
    pars.Reion.use_optical_depth = True
    pars.Reion.reionization = camb.reionization.ReionizationModel.Exponential
    pars.Reion.exponent = 2.0

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=100)
    EE = powers['total'][:,2]  # EE spectrum in muK^2

    # l values: CAMB returns from l=0, so slice from l=2 to l=100
    lmin = 2
    lmax = 100
    ell = np.arange(lmin, lmax+1)
    cl_ee = EE[lmin:lmax+1]  # EE[l] for l in [2,100]

    # Compute l(l+1)C_l^{EE}/(2pi)
    factor = ell * (ell + 1) / (2.0 * np.pi)
    cl_ee_scaled = factor * cl_ee  # [muK^2]

    # Prepare DataFrame
    df = pd.DataFrame({'l': ell, 'EE': cl_ee_scaled})

    # Ensure data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save to CSV
    csv_path = os.path.join(data_dir, "result.csv")
    df.to_csv(csv_path, index=False)

    # Print results to console
    pd.set_option("display.max_rows", None)
    pd.set_option("display.precision", 8)
    print("CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) [muK^2] for l=2 to l=100:")
    print(df)

if __name__ == "__main__":
    compute_cmb_ee_spectrum()
