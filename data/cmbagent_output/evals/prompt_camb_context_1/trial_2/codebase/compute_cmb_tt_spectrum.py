# filename: codebase/compute_cmb_tt_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_tt_spectrum():
    r"""
    Compute the lensed CMB temperature power spectrum (TT) for a non-flat Lambda CDM cosmology
    using CAMB, with the following parameters:
        - Hubble constant (H0): 67.3 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Neutrino mass sum (mnu): 0.06 eV
        - Curvature (omk): 0.05
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965

    The function computes the lensed temperature power spectrum D_l = l(l+1)C_l^{TT}/(2pi) in units of muK^2
    for multipole moments l=2 to l=3000, and saves the results in a CSV file 'data/result.csv' with columns:
        - l: Multipole moment (integer, 2 to 3000)
        - TT: Temperature power spectrum (muK^2)

    Returns
    -------
    None
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.3, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0.05, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965)
    pars.WantScalars = True
    pars.WantTensors = False
    pars.DoLensing = True
    pars.set_for_lmax(lmax=3000, lens_potential_accuracy=1)

    # Run CAMB
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra (D_l = l(l+1)C_l/(2pi), units: muK^2)
    powers = results.get_lensed_scalar_cls(CMB_unit='muK', lmax=3000)
    # powers shape: (lmax+1, 4), columns: TT, EE, BB, TE

    # Extract TT spectrum for l=2 to l=3000
    all_ls = np.arange(powers.shape[0])  # l = 0 ... lmax
    l_values = all_ls[2:3001]            # l = 2 ... 3000
    TT_spectrum = powers[2:3001, 0]      # TT column

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'l': l_values.astype(int), 'TT': TT_spectrum})
    df.to_csv(output_path, index=False)

    # Print summary
    print("CMB lensed temperature power spectrum (TT) computed for non-flat Lambda CDM cosmology.")
    print("Parameters:")
    print("  H0 = 67.3 km/s/Mpc")
    print("  ombh2 = 0.022")
    print("  omch2 = 0.122")
    print("  mnu = 0.06 eV")
    print("  omk = 0.05")
    print("  tau = 0.06")
    print("  As = 2e-9")
    print("  ns = 0.965")
    print("Multipole range: l = 2 to 3000")
    print("Output columns:")
    print("  l  : Multipole moment (integer)")
    print("  TT : l(l+1)C_l^{TT}/(2pi) [muK^2]")
    print("Results saved to " + output_path)
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())


if __name__ == "__main__":
    compute_cmb_tt_spectrum()