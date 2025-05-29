# filename: codebase/camb_power_spectrum.py
import os
import numpy as np
import pandas as pd

# Ensure output directory exists
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    import camb
    from camb import model, initialpower
except ImportError as e:
    print("CAMB is required for this script. Please install it and try again.")
    raise e


def compute_cmb_tt_spectrum():
    r"""
    Computes the CMB temperature power spectrum (C_l^{TT}) for the specified flat Lambda CDM cosmology.

    Returns
    -------
    l_vals : numpy.ndarray
        Array of multipole moments (l), shape (N,), units: dimensionless.
    cl_tt : numpy.ndarray
        Array of temperature power spectrum values (C_l^{TT}), shape (N,), units: microkelvin^2 (uK^2).
    """
    # Cosmological parameters
    H0 = 70.0  # Hubble constant, units: km/s/Mpc
    ombh2 = 0.022  # Baryon density, units: dimensionless (Omega_b h^2)
    omch2 = 0.122  # Cold dark matter density, units: dimensionless (Omega_c h^2)
    mnu = 0.06  # Neutrino mass sum, units: eV
    omk = 0.0  # Curvature, units: dimensionless
    tau = 0.06  # Optical depth to reionization, units: dimensionless
    As = 2e-9  # Scalar amplitude, units: dimensionless
    ns = 0.965  # Scalar spectral index, units: dimensionless

    # Multipole range
    lmin = 2
    lmax = 3000

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.WantTensors = False

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    # 'total' contains the raw C_l's (unlensed, no foregrounds)
    cl_total = powers['total']  # shape: (lmax+1, 4), columns: TT, EE, BB, TE

    # Extract l and TT
    l_vals = np.arange(lmin, lmax+1)  # l=2 to l=3000
    cl_tt = cl_total[lmin:lmax+1, 0]  # TT spectrum, units: (muK)^2

    return l_vals, cl_tt


def save_spectrum_to_csv(l_vals, cl_tt, filename):
    r"""
    Saves the multipole moments and TT power spectrum to a CSV file.

    Parameters
    ----------
    l_vals : numpy.ndarray
        Array of multipole moments (l), shape (N,), units: dimensionless.
    cl_tt : numpy.ndarray
        Array of temperature power spectrum values (C_l^{TT}), shape (N,), units: microkelvin^2 (uK^2).
    filename : str
        Output CSV filename.
    """
    df = pd.DataFrame({'l': l_vals, 'TT': cl_tt})
    df.to_csv(filename, index=False)
    print("Saved CMB TT power spectrum to " + filename)
    # Print summary statistics
    pd.set_option("display.precision", 6)
    print("\nFirst 5 rows of the result:")
    print(df.head())
    print("\nLast 5 rows of the result:")
    print(df.tail())
    print("\nTT spectrum min: " + str(np.min(cl_tt)) + " uK^2, max: " + str(np.max(cl_tt)) + " uK^2")


if __name__ == "__main__":
    l_vals, cl_tt = compute_cmb_tt_spectrum()
    output_file = os.path.join("data", "result.csv")
    save_spectrum_to_csv(l_vals, cl_tt, output_file)