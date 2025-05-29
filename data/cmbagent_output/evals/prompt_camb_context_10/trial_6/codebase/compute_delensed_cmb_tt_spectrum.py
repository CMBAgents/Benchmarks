# filename: codebase/compute_delensed_cmb_tt_spectrum.py
import camb
import numpy as np
import os
import pandas as pd

def compute_and_save_delensed_cmb_tt_spectrum():
    r"""
    Compute the delensed CMB temperature power spectrum D_l^{TT} = l(l+1)C_l^{TT}/(2pi)
    for a flat Lambda CDM cosmology using CAMB, with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Neutrino mass sum (mnu): 0.06 eV
        - Curvature (omk): 0
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965
    The delensed temperature power spectrum is computed for multipole moments l=2 to l=3000,
    applying a delensing efficiency of 80% (i.e., only 20% of the lensing potential remains).
    The results are saved in 'data/result.csv' with columns:
        - l: Multipole moment (integer, 2 to 3000)
        - TT: Delensed temperature power spectrum (D_l^{TT} in muK^2)
    """
    # Cosmological parameters (units in comments)
    H0 = 67.5  # km/s/Mpc
    ombh2 = 0.022  # Omega_b h^2
    omch2 = 0.122  # Omega_c h^2
    mnu = 0.06  # eV
    omk = 0.0  # Flat universe
    tau = 0.06  # Optical depth
    As = 2.0e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index

    # Delensing efficiency and Alens parameter
    delensing_efficiency = 0.8  # 80% of lensing removed
    Alens_param = 1.0 - delensing_efficiency  # 0.2: 20% lensing potential remains

    # Multipole range
    lmin = 2
    lmax_out = 3000  # Output up to l=3000
    lmax_calc = 3500  # Internal calculation lmax for accuracy

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax=lmax_calc, lens_potential_accuracy=1)
    pars.WantScalars = True
    pars.DoLensing = True
    pars.WantTensors = False

    # Run CAMB and get results
    results = camb.get_results(pars)

    # Get partially delensed CMB power spectra
    # Alens=0.2 for 80% delensing efficiency
    # CMB_unit='muK' for output in muK^2
    # raw_cl=False returns D_l = l(l+1)C_l/(2pi)
    delensed_cls = results.get_partially_lensed_cls(
        Alens=Alens_param, lmax=lmax_out, CMB_unit='muK', raw_cl=False
    )

    # Extract TT spectrum for l=2 to l=3000
    ls = np.arange(lmin, lmax_out + 1)
    dl_TT_delensed = delensed_cls[lmin:lmax_out + 1, 0]  # D_l^{TT} in muK^2

    # Prepare output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to CSV
    output_filename = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({"l": ls, "TT": dl_TT_delensed})
    df.to_csv(output_filename, index=False)

    # Print summary to console
    np.set_printoptions(precision=6, suppress=True)
    print("Delensed CMB TT power spectrum (D_l^{TT} = l(l+1)C_l^{TT}/(2pi)) computed and saved.")
    print("Cosmological parameters:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  ombh2 = " + str(ombh2))
    print("  omch2 = " + str(omch2))
    print("  mnu = " + str(mnu) + " eV")
    print("  omk = " + str(omk))
    print("  tau = " + str(tau))
    print("  As = " + str(As))
    print("  ns = " + str(ns))
    print("Delensing efficiency: " + str(delensing_efficiency * 100) + "% (Alens = " + str(Alens_param) + ")")
    print("Multipole range: l = " + str(lmin) + " to " + str(lmax_out))
    print("Output file: " + output_filename)
    print("First 5 rows of the result:")
    print(df.head().to_string(index=False))
    print("Last 5 rows of the result:")
    print(df.tail().to_string(index=False))


if __name__ == "__main__":
    compute_and_save_delensed_cmb_tt_spectrum()