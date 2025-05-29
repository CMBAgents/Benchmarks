# filename: codebase/camb_cmb_ee.py
import os
import numpy as np
import pandas as pd

import camb
from camb import model, initialpower

def compute_cmb_ee_spectrum():
    r"""
    Computes the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for a flat Lambda CDM cosmology
    using CAMB, with specified cosmological parameters. The result is saved as a CSV file in the 'data/' directory.

    Returns
    -------
    None
    """
    # Ensure data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density Omega_b h^2
    omch2 = 0.122  # Cold dark matter density Omega_c h^2
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature Omega_k
    tau = 0.04  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index

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
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total'])

    # Extract EE spectrum
    totCL = powers['total']  # Columns: TT, EE, BB, TE, [PP, TP] if lensing
    # totCL shape: (lmax+1, ncols)
    # EE is column 1

    # l values: 0 to lmax
    ell = np.arange(totCL.shape[0])  # 0 to lmax
    EE = totCL[:, 1]  # EE spectrum in muK^2

    # Compute l(l+1)C_l^{EE}/(2pi)
    factor = ell * (ell + 1) / (2.0 * np.pi)
    EE_power = factor * EE  # muK^2

    # Select l=2 to l=3000
    mask = (ell >= lmin) & (ell <= lmax)
    ell_out = ell[mask]
    EE_out = EE_power[mask]

    # Save to CSV
    df = pd.DataFrame({'l': ell_out, 'EE': EE_out})
    output_path = os.path.join(data_dir, "result.csv")
    df.to_csv(output_path, index=False)

    # Print summary
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print("CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) computed for l=2 to l=3000.")
    print("Units: microkelvin^2 (uK^2)")
    print("Results saved to: " + output_path)
    print("First 10 rows:")
    print(df.head(10).to_string(index=False))
    print("Last 10 rows:")
    print(df.tail(10).to_string(index=False))


if __name__ == "__main__":
    compute_cmb_ee_spectrum()
