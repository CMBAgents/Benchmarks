# filename: codebase/compute_cmb_tt_spectrum.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

def compute_cmb_tt_spectrum():
    r"""
    Compute the CMB temperature power spectrum (C_l^{TT}) for a flat Lambda CDM cosmology
    using CAMB with the specified parameters. The results are saved in 'data/result.csv'.

    Parameters (units in comments):
        H0: Hubble constant [km/s/Mpc]
        ombh2: Baryon density parameter [dimensionless]
        omch2: Cold dark matter density parameter [dimensionless]
        mnu: Sum of neutrino masses [eV]
        omk: Curvature parameter [dimensionless]
        tau: Optical depth to reionization [dimensionless]
        As: Scalar amplitude [dimensionless]
        ns: Scalar spectral index [dimensionless]

    Output:
        CSV file with columns:
            l: Multipole moment (integer, 2 to 3000)
            TT: Raw temperature power spectrum [μK^2]
    """
    # Cosmological parameters
    H0 = 74.0  # km/s/Mpc
    ombh2 = 0.022  # Omega_b h^2
    omch2 = 0.122  # Omega_c h^2
    mnu = 0.06  # eV
    omk = 0.0  # Flat universe
    tau = 0.06  # Optical depth
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
    tt = powers['unlensed_scalar'][lmin:lmax+1, 0]  # TT spectrum, μK^2

    # Prepare data for saving
    l_vals = np.arange(lmin, lmax+1)
    df = pd.DataFrame({'l': l_vals, 'TT': tt})

    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    df.to_csv(output_path, index=False)

    # Print summary of results
    pd.set_option("display.precision", 8)
    pd.set_option("display.max_rows", 10)
    print("CMB TT power spectrum (μK^2) saved to " + output_path)
    print("First and last few rows of the result:")
    print(pd.concat([df.head(5), df.tail(5)]))


if __name__ == "__main__":
    compute_cmb_tt_spectrum()