# filename: codebase/cmb_tt_spectrum.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

def compute_cmb_tt_spectrum():
    r"""
    Compute the CMB temperature power spectrum (C_l^{TT}) for the specified cosmology
    and save the results to 'data/result.csv'.

    Returns
    -------
    None
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cosmological parameters
    H0 = 74.0  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Omega_b h^2 (baryon density)
    omch2 = 0.122  # Omega_c h^2 (cold dark matter density)
    mnu = 0.06  # sum m_nu [eV]
    omk = 0.0  # Omega_k (curvature)
    tau = 0.06  # Optical depth to reionization
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
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
    tt = powers['unlensed_scalar'][:, 0]  # TT spectrum (μK^2)

    # l values corresponding to the spectrum
    ell = np.arange(tt.size)
    # Select l=2 to l=3000
    mask = (ell >= lmin) & (ell <= lmax)
    ell = ell[mask]
    tt = tt[mask]

    # Save to CSV
    df = pd.DataFrame({'l': ell, 'TT': tt})
    csv_path = os.path.join(output_dir, "result.csv")
    df.to_csv(csv_path, index=False)

    # Print summary
    print("CMB TT power spectrum computed for l = 2 to 3000.")
    print("Results saved to " + csv_path)
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())
    print("TT units: microkelvin^2 (μK^2)")

if __name__ == "__main__":
    compute_cmb_tt_spectrum()
