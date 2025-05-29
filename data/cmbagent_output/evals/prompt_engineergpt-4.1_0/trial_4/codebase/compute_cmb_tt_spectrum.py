# filename: codebase/compute_cmb_tt_spectrum.py
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
    print("CAMB package is required but not installed. Please install camb and rerun the code.")
    raise e

def compute_cmb_tt_spectrum():
    r"""
    Computes the CMB temperature power spectrum for the specified flat Lambda CDM cosmology.

    Returns:
        l_vals (np.ndarray): Multipole moments (l) [unitless]
        tt_spectrum (np.ndarray): Temperature power spectrum (l(l+1)C_l^{TT}/(2π)) [unit: μK²]
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.02  # Baryon density [unitless]
    omch2 = 0.122  # Cold dark matter density [unitless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [unitless]
    tau = 0.06  # Optical depth [unitless]
    As = 2e-9  # Scalar amplitude [unitless]
    ns = 0.965  # Scalar spectral index [unitless]

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(3000, lens_potential_accuracy=0)
    pars.WantTensors = False

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000)
    totCL = powers['total']

    # l values: CAMB returns from l=0,1,...,lmax
    l_vals = np.arange(totCL.shape[0])  # l=0 to lmax
    # Extract l=2 to l=3000
    lmin = 2
    lmax = 3000
    l_range = np.arange(lmin, lmax + 1)
    # TT spectrum: column 0 is TT
    cl_tt = totCL[lmin:lmax + 1, 0]  # [μK²]

    # Compute l(l+1)C_l/(2π) [μK²]
    tt_spectrum = l_range * (l_range + 1) * cl_tt / (2.0 * np.pi)

    return l_range, tt_spectrum

def save_spectrum_to_csv(l_vals, tt_spectrum, filename):
    r"""
    Saves the CMB TT power spectrum to a CSV file.

    Args:
        l_vals (np.ndarray): Multipole moments [unitless]
        tt_spectrum (np.ndarray): TT power spectrum [μK²]
        filename (str): Output CSV filename
    """
    df = pd.DataFrame({'l': l_vals, 'TT': tt_spectrum})
    df.to_csv(filename, index=False)
    print("CMB TT power spectrum saved to " + filename)
    print("First 10 rows:")
    print(df.head(10).to_string(index=False))
    print("\nLast 10 rows:")
    print(df.tail(10).to_string(index=False))
    print("\nTotal number of multipoles: " + str(len(df)))


if __name__ == "__main__":
    l_vals, tt_spectrum = compute_cmb_tt_spectrum()
    save_spectrum_to_csv(l_vals, tt_spectrum, "data/result.csv")
