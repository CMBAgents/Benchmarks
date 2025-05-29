# filename: codebase/cmb_bb_power_spectrum.py
import os
import numpy as np
import pandas as pd

# Ensure data directory exists
database_path = "data"
if not os.path.exists(database_path):
    os.makedirs(database_path)

try:
    import camb
    from camb import model, initialpower
except ImportError as e:
    print("CAMB is not installed. Please install camb and rerun the code.")
    raise e


def compute_cmb_bb_spectrum():
    r"""
    Computes the CMB B-mode polarization power spectrum (l(l+1)C_l^{BB}/(2pi)) for the specified cosmology.
    Returns:
        l_vals (np.ndarray): Multipole moments (l), shape (N,), unitless
        BB_spectrum (np.ndarray): B-mode power spectrum (micro-Kelvin^2), shape (N,), units: micro-Kelvin^2
    """
    # Set up CAMB parameters
    pars = camb.CAMBparams()
    # Set cosmological parameters
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0.0, tau=0.06)
    # Set initial power spectrum parameters
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    # Set lmax
    lmax = 3000
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    # Only need the CMB power spectra
    pars.WantTensors = True  # Needed for BB, even if r=0
    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total'])
    # Extract BB spectrum
    cl = powers['total']  # columns: TT, EE, BB, TE, [PP, TP, EP] if lensing
    # l values: cl[0] is l=0, cl[1] is l=1, ...
    l_vals = np.arange(cl.shape[0])
    # Only keep l=2 to l=3000
    lmin = 2
    lmax = 3000
    l_vals = l_vals[lmin:lmax+1]
    # BB is column 2
    cl_BB = cl[lmin:lmax+1, 2]  # units: micro-Kelvin^2
    # Compute l(l+1)C_l^{BB}/(2pi)
    BB_spectrum = l_vals * (l_vals + 1) * cl_BB / (2.0 * np.pi)
    return l_vals, BB_spectrum


def save_results_to_csv(l_vals, BB_spectrum, filename):
    r"""
    Save the B-mode power spectrum to a CSV file.
    Args:
        l_vals (np.ndarray): Multipole moments (l), shape (N,)
        BB_spectrum (np.ndarray): B-mode power spectrum (micro-Kelvin^2), shape (N,)
        filename (str): Output CSV file path
    """
    df = pd.DataFrame({'l': l_vals.astype(int), 'BB': BB_spectrum})
    df.to_csv(filename, index=False)
    # Print summary
    print("Saved B-mode power spectrum to " + filename)
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())
    print("Total number of multipoles: " + str(len(df)))
    print("BB spectrum units: micro-Kelvin^2")


if __name__ == "__main__":
    l_vals, BB_spectrum = compute_cmb_bb_spectrum()
    output_csv = os.path.join(database_path, "result.csv")
    save_results_to_csv(l_vals, BB_spectrum, output_csv)
