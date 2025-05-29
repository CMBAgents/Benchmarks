# filename: codebase/compute_delensed_cmb_tt_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_delensed_cmb_tt_spectrum():
    r"""
    Compute the delensed CMB temperature power spectrum l(l+1)C_l^{TT}/(2pi) for a flat Lambda CDM cosmology
    using CAMB, with specified cosmological parameters and 80% delensing efficiency.

    The result is saved as a CSV file 'data/result.csv' with columns:
        l: Multipole moment (integer, 2 to 3000)
        TT: Delensed temperature power spectrum (l(l+1)C_l^{TT}/(2pi)) in microkelvin squared (muK^2)

    Units:
        - l: dimensionless
        - TT: microkelvin squared (muK^2)
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cosmological parameters
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

    # Set calculation lmax and lensing accuracy
    lmax_calc = 3500  # Calculation lmax (should be > output lmax for accuracy)
    pars.set_for_lmax(lmax=lmax_calc, lens_potential_accuracy=1)

    # Run CAMB
    results = camb.get_results(pars)

    # Delensing efficiency: 80% (i.e., 80% of lensing removed, so Alens=0.2)
    delensing_efficiency = 0.8
    Alens_residual = 1.0 - delensing_efficiency  # Fraction of lensing potential remaining

    # Get delensed CMB power spectra (lensed, with reduced lensing)
    # Output lmax = 3000, units = muK^2, output is l(l+1)C_l/2pi
    lmax_out = 3000
    cls = results.get_partially_lensed_cls(Alens=Alens_residual, lmax=lmax_out, CMB_unit='muK', raw_cl=False)

    # Extract multipole moments and TT spectrum
    ls = np.arange(2, lmax_out + 1)  # l = 2..3000, dimensionless
    TT = cls[2:lmax_out + 1, 0]  # TT spectrum, muK^2

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'l': ls, 'TT': TT})
    df.to_csv(output_path, index=False)

    # Print summary to console
    print("Delensed CMB temperature power spectrum (l(l+1)C_l^{TT}/(2pi)) computed for l=2..3000.")
    print("Cosmological parameters used:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  Sum m_nu = " + str(mnu) + " eV")
    print("  Omega_k = " + str(omk))
    print("  tau = " + str(tau))
    print("  A_s = " + str(As))
    print("  n_s = " + str(ns))
    print("Delensing efficiency: 80% (Alens = 0.2)")
    print("Results saved to " + output_path)
    print("First 5 rows of the result:")
    print(df.head())


if __name__ == "__main__":
    compute_delensed_cmb_tt_spectrum()