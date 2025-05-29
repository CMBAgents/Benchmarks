# filename: codebase/cmb_ee_spectrum.py
import camb
import numpy as np
import math
import os
import pandas as pd

def compute_cmb_ee_spectrum():
    r"""
    Compute the CMB E-mode polarization power spectrum (Dl^EE = l(l+1)C_l^{EE}/2pi) for a flat Lambda CDM cosmology
    using CAMB, with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Scalar amplitude (As): 1.8e-9 * exp(2 * tau)
        - Scalar spectral index (ns): 0.95
        - Optical depth to reionization (tau): 0.1
        - Reionization model: Exponential reionization with exponent power 2

    The function computes the lensed E-mode power spectrum in units of micro-Kelvin squared (muK^2) for multipole
    moments l=2 to l=100, and saves the results in a CSV file 'data/result.csv' with columns:
        - l: Multipole moment (integer, 2 to 100)
        - EE: E-mode polarization power spectrum (Dl^EE, muK^2)

    Returns
    -------
    None
    """
    # Cosmological parameters (units in comments)
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    tau = 0.1  # Optical depth to reionization [dimensionless]
    As_base = 1.8e-9  # Scalar amplitude base [dimensionless]
    As = As_base * math.exp(2 * tau)  # Scalar amplitude [dimensionless]
    ns = 0.95  # Scalar spectral index [dimensionless]
    reion_exp_power = 2  # Exponent for exponential reionization [dimensionless]
    omk = 0.0  # Flat universe [dimensionless]

    # Multipole range
    lmin = 2
    lmax = 100  # Maximum multipole for output [dimensionless]
    lmax_calc = 200  # Calculation lmax for accuracy [dimensionless]
    lens_potential_accuracy = 1  # For lensed spectra [dimensionless]

    # Set up CAMB parameters
    pars = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        omk=omk,
        tau=tau,
        As=As,
        ns=ns,
        reionization_model='ExpReionization',
        reion_exp_power=reion_exp_power,
        WantScalars=True,
        WantTensors=False,
        lmax=lmax_calc,
        lens_potential_accuracy=lens_potential_accuracy
    )

    # Compute results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in muK^2, Dl = l(l+1)Cl/(2pi)
    powers = results.get_cmb_power_spectra(
        pars,
        CMB_unit='muK',
        spectra=['lensed_scalar'],
        lmax=lmax
    )
    # powers['lensed_scalar'] shape: (lmax+1, 4), columns: TT, EE, BB, TE
    Dl_EE = powers['lensed_scalar'][:, 1]  # EE column, muK^2

    # Prepare output for l=2..100
    l_arr = np.arange(lmin, lmax + 1)
    EE_arr = Dl_EE[lmin:lmax + 1]

    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to CSV
    output_csv = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'l': l_arr, 'EE': EE_arr})
    df.to_csv(output_csv, index=False)

    # Print summary to console
    print("CMB E-mode polarization power spectrum (Dl^EE = l(l+1)C_l^{EE}/2pi) computed for:")
    print("  H0 = " + str(H0) + " km/s/Mpc, ombh2 = " + str(ombh2) + ", omch2 = " + str(omch2) + ", tau = " + str(tau))
    print("  As = " + str(As) + ", ns = " + str(ns) + ", reionization: ExpReionization (power = " + str(reion_exp_power) + ")")
    print("  l range: " + str(lmin) + " to " + str(lmax))
    print("Results saved to: " + output_csv)
    print("First 5 rows:")
    print(df.head())


if __name__ == "__main__":
    compute_cmb_ee_spectrum()
