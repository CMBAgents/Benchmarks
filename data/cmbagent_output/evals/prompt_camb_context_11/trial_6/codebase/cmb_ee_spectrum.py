# filename: codebase/cmb_ee_spectrum.py
import os
import math
import numpy as np
import pandas as pd
import camb
from camb import model

def compute_cmb_ee_spectrum(
    H0=67.5,                # [km/s/Mpc]
    ombh2=0.022,            # [dimensionless]
    omch2=0.122,            # [dimensionless]
    ns=0.95,                # [dimensionless]
    tau=0.1,                # [dimensionless]
    reion_exp_power=2,      # [dimensionless]
    lmax_calc=500,          # [dimensionless]
    lmin_output=2,          # [dimensionless]
    lmax_output=100,        # [dimensionless]
    output_dir="data/",
    output_filename="result.csv"
):
    r"""
    Compute the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for a flat Lambda CDM cosmology
    using CAMB, with specified cosmological and reionization parameters.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Baryon density Omega_b h^2 (dimensionless).
    omch2 : float
        Cold dark matter density Omega_c h^2 (dimensionless).
    ns : float
        Scalar spectral index (dimensionless).
    tau : float
        Optical depth to reionization (dimensionless).
    reion_exp_power : float
        Exponent power for exponential reionization model (dimensionless).
    lmax_calc : int
        Maximum multipole for accurate calculation (dimensionless).
    lmin_output : int
        Minimum multipole to output (dimensionless).
    lmax_output : int
        Maximum multipole to output (dimensionless).
    output_dir : str
        Directory to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
        Saves the results to a CSV file and prints a summary to the console.
    """
    # Calculate scalar amplitude As = 1.8e-9 * exp(2 * tau)
    As = 1.8e-9 * math.exp(2.0 * tau)  # [dimensionless]

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up CAMB parameters
    pars = camb.model.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=0.0, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    # Fixed reionization model setting
    pars.Reion.ReionizationModel = 'ExpReionization'
    pars.Reion.set_extra_params(reion_exp_power=reion_exp_power)
    pars.set_for_lmax(lmax=lmax_calc, lens_potential_accuracy=1)
    pars.WantScalars = True
    pars.WantTensors = False

    # Run CAMB and get results
    results = camb.get_results(pars)
    # Get lensed scalar CMB power spectra in muK^2, up to lmax_output
    cls = results.get_lensed_scalar_cls(lmax=lmax_output, CMB_unit='muK')

    # Extract l and EE spectrum for l=lmin_output to lmax_output
    ls = np.arange(lmin_output, lmax_output + 1)  # [dimensionless]
    EE = cls[lmin_output:lmax_output + 1, 1]      # [muK^2]

    # Save to CSV
    output_path = os.path.join(output_dir, output_filename)
    df = pd.DataFrame({'l': ls, 'EE': EE})
    df.to_csv(output_path, index=False)

    # Print detailed summary
    print("CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) computed for:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  n_s = " + str(ns))
    print("  tau = " + str(tau))
    print("  A_s = " + str(As))
    print("  Reionization model: Exponential, exponent power = " + str(reion_exp_power))
    print("  l range: " + str(lmin_output) + " to " + str(lmax_output))
    print("Results saved to: " + output_path)
    print("First 10 rows of the output (l, EE [muK^2]):")
    with pd.option_context('display.max_rows', 10, 'display.max_columns', 2, 'display.width', 120):
        print(df.head(10))


if __name__ == "__main__":
    compute_cmb_ee_spectrum()