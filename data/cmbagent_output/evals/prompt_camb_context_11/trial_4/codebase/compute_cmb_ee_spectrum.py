# filename: codebase/compute_cmb_ee_spectrum.py
import camb
import numpy as np
import math
import os
import pandas as pd

def compute_cmb_ee_spectrum():
    r"""
    Compute the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for a flat Lambda CDM cosmology
    using CAMB, with specified cosmological and reionization parameters.

    Returns
    -------
    l_vals : np.ndarray
        Multipole moments (l), shape (99,), unitless (integer values from 2 to 100)
    EE_vals : np.ndarray
        E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)), shape (99,), units: muK^2
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [unitless]
    omch2 = 0.122  # Cold dark matter density [unitless]
    tau = 0.1  # Optical depth to reionization [unitless]
    ns = 0.95  # Scalar spectral index [unitless]
    As = 1.8e-9 * math.exp(2 * tau)  # Scalar amplitude [unitless]
    omk = 0.0  # Flat universe [unitless]
    reion_exp_power = 2  # Exponential reionization exponent [unitless]
    lmax = 100  # Maximum multipole moment for output [unitless]
    lens_potential_accuracy = 1  # Lensing accuracy [unitless]

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
        lmax=lmax,
        lens_potential_accuracy=lens_potential_accuracy
    )

    # Compute results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in muK^2, Dl form (l(l+1)C_l/2pi)
    powers = results.get_cmb_power_spectra(
        pars,
        CMB_unit='muK',
        spectra=['lensed_scalar'],
        lmax=lmax
    )

    # Extract EE spectrum (column 1), l=2..100
    cl_EE_Dl = powers['lensed_scalar'][:, 1]  # EE is column 1, shape (lmax+1,)
    l_vals = np.arange(2, lmax + 1)  # l=2..100, shape (99,)
    EE_vals = cl_EE_Dl[2:lmax + 1]  # shape (99,)

    return l_vals, EE_vals

def save_to_csv(l_vals, EE_vals, filename):
    r"""
    Save the multipole moments and E-mode power spectrum to a CSV file.

    Parameters
    ----------
    l_vals : np.ndarray
        Multipole moments (l), shape (N,), unitless
    EE_vals : np.ndarray
        E-mode polarization power spectrum (muK^2), shape (N,)
    filename : str
        Output CSV filename
    """
    df = pd.DataFrame({'l': l_vals, 'EE': EE_vals})
    df.to_csv(filename, index=False)
    print("Saved CMB E-mode power spectrum to " + filename)
    print("First 5 rows of the saved data:")
    print(df.head())
    print("Last 5 rows of the saved data:")
    print(df.tail())
    print("Total number of rows saved: " + str(len(df)))


if __name__ == "__main__":
    # Ensure data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Compute spectrum
    l_vals, EE_vals = compute_cmb_ee_spectrum()

    # Save to CSV
    output_csv = os.path.join(data_dir, "result.csv")
    save_to_csv(l_vals, EE_vals, output_csv)