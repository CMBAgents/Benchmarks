# filename: codebase/compute_cmb_tt_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_tt_spectrum():
    r"""
    Compute the raw (unlensed) CMB temperature power spectrum (C_l^{TT}) for a flat Lambda CDM cosmology
    using CAMB with the following parameters:
        - Hubble constant (H0): 70 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Neutrino mass sum (mnu): 0.06 eV
        - Curvature (omk): 0
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965

    The function computes the temperature power spectrum (C_l^{TT}) in units of μK^2 for multipole moments
    from l=2 to l=3000 and saves the results in a CSV file named 'result.csv' with two columns:
        - l: Multipole moment (integer values from 2 to 3000)
        - TT: Temperature power spectrum (C_l^{TT} in μK^2)

    The CSV file is saved in the 'data/' directory.
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=70, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965)
    pars.set_for_lmax(lmax=3000, lens_potential_accuracy=0)
    pars.WantScalars = True
    pars.WantTensors = False
    pars.WantVectors = False
    pars.WantCls = True
    pars.DoLensing = False

    # Run CAMB and get results
    results = camb.get_results(pars)

    # Get the unlensed scalar CMB power spectra (in μK^2, raw Cl)
    lmax_output = 3000
    unlensed_cls = results.get_unlensed_scalar_cls(CMB_unit='muK', raw_cl=True, lmax=lmax_output)

    # Extract TT spectrum for l=2 to 3000
    ls = np.arange(2, lmax_output + 1)  # l values
    cl_TT = unlensed_cls[2:lmax_output + 1, 0]  # TT spectrum in μK^2

    # Save to CSV
    df = pd.DataFrame({'l': ls, 'TT': cl_TT})
    csv_path = os.path.join(output_dir, "result.csv")
    df.to_csv(csv_path, index=False)

    # Print detailed information
    print("CMB TT power spectrum (unlensed, raw Cl) computed for flat Lambda CDM cosmology.")
    print("Cosmological parameters used:")
    print("  H0 = 70 km/s/Mpc")
    print("  ombh2 = 0.022")
    print("  omch2 = 0.122")
    print("  mnu = 0.06 eV")
    print("  omk = 0")
    print("  tau = 0.06")
    print("  As = 2e-9")
    print("  ns = 0.965")
    print("Multipole range: l = 2 to 3000")
    print("Results saved to: " + csv_path)
    print("First 5 rows of the result:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    compute_cmb_tt_spectrum()