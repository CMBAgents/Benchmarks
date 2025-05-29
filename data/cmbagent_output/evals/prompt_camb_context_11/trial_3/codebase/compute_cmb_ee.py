# filename: codebase/compute_cmb_ee.py
r"""
Compute the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for a flat Lambda CDM cosmology
using CAMB, with the following parameters:
- Hubble constant (H0): 67.5 km/s/Mpc
- Baryon density (ombh2): 0.022
- Cold dark matter density (omch2): 0.122
- Scalar amplitude (As): 1.8e-9 * exp(2 * tau)
- Scalar spectral index (ns): 0.95
- Optical depth to reionization (tau): 0.1
- Reionization model: Exponential reionization with exponent power 2

The E-mode power spectrum (l(l+1)C_l^{EE}/(2pi)) is computed in units of micro-Kelvin squared (uK^2)
for multipole moments l=2 to l=100, and saved to 'data/result.csv' with columns:
'l' (multipole moment), 'EE' (E-mode power spectrum in uK^2).

All units are SI except where otherwise noted.
"""

import camb
import numpy as np
import math
import os
import pandas as pd


def compute_cmb_ee_spectrum(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density [dimensionless]
    omch2=0.122,            # Cold dark matter density [dimensionless]
    tau=0.1,                # Optical depth to reionization [dimensionless]
    As_base=1.8e-9,         # Scalar amplitude base [dimensionless]
    ns=0.95,                # Scalar spectral index [dimensionless]
    reion_exp_power=2,      # Exponential reionization exponent [dimensionless]
    lmin=2,                 # Minimum multipole [dimensionless]
    lmax=100,               # Maximum multipole [dimensionless]
    output_csv='data/result.csv' # Output CSV file path
):
    r"""
    Compute the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for a flat Lambda CDM cosmology
    using CAMB, and save the results to a CSV file.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter.
    omch2 : float
        Physical cold dark matter density parameter.
    tau : float
        Optical depth to reionization.
    As_base : float
        Base scalar amplitude (A_s).
    ns : float
        Scalar spectral index.
    reion_exp_power : float
        Exponent for exponential reionization model.
    lmin : int
        Minimum multipole moment to include in output.
    lmax : int
        Maximum multipole moment to include in output.
    output_csv : str
        Path to output CSV file.

    Returns
    -------
    None
    """
    # Calculate scalar amplitude As = As_base * exp(2 * tau)
    As = As_base * math.exp(2.0 * tau)

    # Set up CAMB parameters
    pars = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        omk=0.0,  # Flat universe
        tau=tau,
        As=As,
        ns=ns,
        reionization_model='ExpReionization',
        reion_exp_power=reion_exp_power,
        WantScalars=True,
        WantTensors=False,
        lmax=lmax,
        lens_potential_accuracy=1
    )

    # Compute results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in muK^2, Dl convention
    powers = results.get_cmb_power_spectra(
        pars,
        CMB_unit='muK',
        spectra=['lensed_scalar'],
        lmax=lmax
    )

    # Extract EE spectrum: powers['lensed_scalar'][:, 1] (index 1 is EE)
    cl_ee_dl = powers['lensed_scalar'][:, 1]  # shape: (lmax+1,)

    # Prepare l and EE arrays for l=lmin to lmax
    l_arr = np.arange(lmin, lmax + 1, dtype=int)
    ee_arr = cl_ee_dl[lmin:lmax + 1]

    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to CSV using pandas for clarity
    df = pd.DataFrame({'l': l_arr, 'EE': ee_arr})
    df.to_csv(output_csv, index=False)

    # Print detailed results to console
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    pd.set_option('display.precision', 6)
    pd.set_option('display.width', 120)
    pd.set_option('display.max_rows', 10)
    print("CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) in micro-Kelvin^2 (uK^2):")
    print("Multipole l range: " + str(lmin) + " to " + str(lmax))
    print("First 10 rows of the result:")
    print(df.head(10))
    print("\nSaved full result to " + output_csv)


if __name__ == "__main__":
    compute_cmb_ee_spectrum()