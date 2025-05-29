# filename: codebase/compute_delensed_cmb_spectrum.py
import camb
import numpy as np
import pandas as pd
import os


def compute_delensed_cmb_tt_spectrum():
    r"""
    Compute the delensed CMB temperature power spectrum l(l+1)C_l^{TT}/(2pi) for a flat Lambda CDM cosmology
    using CAMB, with specified cosmological parameters and 80% delensing efficiency.

    The result is saved as a CSV file with columns:
        - l: Multipole moment (integer, 2 to 3000)
        - TT: Delensed temperature power spectrum (l(l+1)C_l^{TT}/(2pi)) in microkelvin squared (muK^2)

    The file is saved as 'data/result.csv'.

    Units:
        - TT: microkelvin squared (muK^2)
        - l: dimensionless

    Returns
    -------
    None
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cosmological parameters (all units as required by CAMB)
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    As = 2e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]

    # Set up CAMB parameters
    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau, As=As, ns=ns)

    # Set calculation lmax higher than output lmax for accuracy
    lmax_calc = 3500  # Calculation lmax (for accuracy)
    lmax_out = 3000   # Output lmax (as required)
    pars.set_for_lmax(lmax=lmax_calc, lens_potential_accuracy=1)

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Delensing efficiency: 80% (i.e., 80% of lensing removed, 20% remains)
    delensing_efficiency = 0.8
    Alens_residual = 1.0 - delensing_efficiency  # Fraction of lensing potential remaining

    # Get the delensed CMB power spectra (in muK^2, lensed, lmax_out)
    # Output: array of shape (lmax_out+1, 4), columns: TT, EE, BB, TE
    cls = results.get_partially_lensed_cls(
        Alens=Alens_residual,
        lmax=lmax_out,
        CMB_unit='muK',
        raw_cl=False  # Returns l(l+1)C_l/2pi
    )

    # Extract multipole moments and TT spectrum
    ls = np.arange(2, lmax_out + 1)  # l = 2..3000
    TT = cls[2:lmax_out + 1, 0]      # TT spectrum, muK^2

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'l': ls, 'TT': TT})
    df.to_csv(output_path, index=False)

    # Print summary to console
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    pd.set_option('display.precision', 6)
    pd.set_option('display.width', 120)
    pd.set_option('display.max_rows', 10)
    print("Delensed CMB TT power spectrum (l(l+1)C_l^{TT}/(2pi)) computed and saved to " + output_path)
    print("Cosmological parameters used:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  ombh2 = " + str(ombh2))
    print("  omch2 = " + str(omch2))
    print("  mnu = " + str(mnu) + " eV")
    print("  omk = " + str(omk))
    print("  tau = " + str(tau))
    print("  As = " + str(As))
    print("  ns = " + str(ns))
    print("Delensing efficiency: " + str(delensing_efficiency * 100.0) + "% (Alens = " + str(Alens_residual) + ")")
    print("Output columns: l (multipole, 2..3000), TT (muK^2)")
    print("First few rows of the result:")
    print(df.head(10))
    print("Total number of rows: " + str(len(df)))


if __name__ == "__main__":
    compute_delensed_cmb_tt_spectrum()