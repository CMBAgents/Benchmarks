# filename: codebase/cmb_bb_spectrum.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

def compute_cmb_bb_spectrum():
    r"""
    Computes the CMB B-mode polarization power spectrum (l(l+1)C_l^{BB}/(2pi)) for a flat Lambda CDM cosmology
    using CAMB, with specified cosmological parameters.

    Returns
    -------
    l : ndarray
        Multipole moments (l), shape (N,)
    BB : ndarray
        B-mode power spectrum (l(l+1)C_l^{BB}/(2pi)) in micro-Kelvin squared (uK^2), shape (N,)
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    r_tensor = 0.0  # Tensor-to-scalar ratio [dimensionless]
    As = 2e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]

    # Multipole range
    lmin = 2
    lmax = 3000

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=r_tensor)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.WantTensors = True  # Needed for BB spectrum, even if r=0

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total'])

    # Extract BB spectrum
    cl = powers['total']  # shape: (lmax+1, 4) for TT, EE, BB, TE
    # cl[:,2] is BB
    BB = cl[:,2]  # BB in muK^2

    # Compute l(l+1)C_l^{BB}/(2pi)
    ell = np.arange(cl.shape[0])  # l=0..lmax
    factor = ell * (ell + 1) / (2.0 * np.pi)
    BB_lensed = factor * BB  # units: muK^2

    # Select l=2..lmax
    l_out = ell[lmin:lmax+1]
    BB_out = BB_lensed[lmin:lmax+1]

    return l_out, BB_out

def save_results_to_csv(l, BB, filename):
    r"""
    Save the multipole moments and B-mode power spectrum to a CSV file.

    Parameters
    ----------
    l : ndarray
        Multipole moments (l), shape (N,)
    BB : ndarray
        B-mode power spectrum (l(l+1)C_l^{BB}/(2pi)) in micro-Kelvin squared (uK^2), shape (N,)
    filename : str
        Output CSV filename
    """
    df = pd.DataFrame({'l': l, 'BB': BB})
    df.to_csv(filename, index=False)
    print("Saved B-mode power spectrum to " + filename)
    print("First 10 rows:")
    print(df.head(10))
    print("Last 10 rows:")
    print(df.tail(10))
    print("Total rows: " + str(len(df)))


if __name__ == "__main__":
    # Ensure data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Compute spectrum
    l, BB = compute_cmb_bb_spectrum()

    # Save to CSV
    output_file = os.path.join(data_dir, "result.csv")
    save_results_to_csv(l, BB, output_file)