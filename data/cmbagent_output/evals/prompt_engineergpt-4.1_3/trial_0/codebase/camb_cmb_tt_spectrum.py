# filename: codebase/camb_cmb_tt_spectrum.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

def compute_cmb_tt_spectrum():
    r"""
    Compute the CMB temperature power spectrum (C_l^{TT}) for a flat Lambda CDM cosmology
    using CAMB with specified parameters. The results are saved in a CSV file with columns:
    l (multipole moment) and TT (temperature power spectrum in μK^2).

    Parameters used:
    - Hubble constant (H0): 74 km/s/Mpc
    - Baryon density (ombh2): 0.022
    - Cold dark matter density (omch2): 0.122
    - Neutrino mass sum (mnu): 0.06 eV
    - Curvature (omk): 0
    - Optical depth to reionization (tau): 0.06
    - Scalar amplitude (As): 2e-9
    - Scalar spectral index (ns): 0.965

    Output:
    - CSV file 'data/result.csv' with columns 'l' and 'TT'
    - Prints a summary of the results to the console
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set cosmological parameters
    H0 = 74.0  # Hubble constant [km/s/Mpc]
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

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
    cl = powers['unlensed_scalar']  # shape: (lmax+1, 4), columns: TT, EE, BB, TE

    # Extract l and TT
    lmin = 2
    lmax = 3000
    ell = np.arange(lmin, lmax + 1)  # Multipole moments
    TT = cl[lmin:lmax + 1, 0]  # TT spectrum in μK^2

    # Save to CSV
    df = pd.DataFrame({'l': ell, 'TT': TT})
    csv_path = os.path.join(output_dir, "result.csv")
    df.to_csv(csv_path, index=False)

    # Print summary of results
    np.set_printoptions(precision=6, suppress=True)
    pd.set_option('display.float_format', lambda x: '%.6f' % x)
    print("CMB temperature power spectrum (TT) computed for l = 2 to 3000.")
    print("Results saved to " + csv_path)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())

if __name__ == "__main__":
    compute_cmb_tt_spectrum()
