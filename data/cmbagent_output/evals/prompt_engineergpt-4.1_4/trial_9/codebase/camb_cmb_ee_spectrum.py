# filename: codebase/camb_cmb_ee_spectrum.py
import os
import numpy as np
import pandas as pd

# Ensure data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

try:
    import camb
    from camb import model, initialpower
except ImportError as e:
    print("CAMB is not installed. Please install the camb package to run this code.")
    raise e


def compute_cmb_ee_spectrum():
    r"""
    Computes the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for the specified cosmology.
    Returns:
        l_vals (np.ndarray): Multipole moments (l), shape (N,), units: dimensionless
        cl_ee (np.ndarray): E-mode power spectrum (microkelvin^2), shape (N,), units: microkelvin^2
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.04  # Optical depth to reionization [dimensionless]
    As = 2e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]

    # Multipole range
    lmax = 3000

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.WantTensors = False
    pars.Want_CMB_lensing = False

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total'])

    # Extract EE spectrum
    totCL = powers['total']  # shape: (lmax+1, 4) [TT, EE, BB, TE]
    # l values start from 0, but C_l is only meaningful for l>=2
    l_vals = np.arange(totCL.shape[0])  # l = 0 ... lmax
    EE = totCL[:, 1]  # EE column, units: microkelvin^2

    # Compute l(l+1)C_l^{EE}/(2pi) for l=2 to lmax
    lmin = 2
    lmax_eff = min(lmax, len(EE)-1)
    l_range = np.arange(lmin, lmax_eff+1)
    cl_ee = EE[lmin:lmax_eff+1] * l_range * (l_range + 1) / (2.0 * np.pi)  # units: microkelvin^2

    return l_range, cl_ee


def save_results_to_csv(l_vals, cl_ee, filename):
    r"""
    Saves the multipole moments and E-mode power spectrum to a CSV file.
    Args:
        l_vals (np.ndarray): Multipole moments (l), shape (N,)
        cl_ee (np.ndarray): E-mode power spectrum (microkelvin^2), shape (N,)
        filename (str): Output CSV filename
    """
    df = pd.DataFrame({'l': l_vals, 'EE': cl_ee})
    df.to_csv(filename, index=False)
    print("Saved E-mode power spectrum to " + filename)
    # Print summary
    pd.set_option("display.precision", 6)
    pd.set_option("display.max_rows", 10)
    print("First and last 5 rows of the result:")
    print(pd.concat([df.head(5), df.tail(5)]))


if __name__ == "__main__":
    l_vals, cl_ee = compute_cmb_ee_spectrum()
    save_results_to_csv(l_vals, cl_ee, "data/result.csv")
