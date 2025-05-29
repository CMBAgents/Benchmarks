# filename: codebase/camb_cmb_ee.py
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
    print("CAMB is not installed. Please install camb and rerun the code.")
    raise e


def compute_cmb_ee_spectrum():
    r"""
    Compute the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for a flat Lambda CDM cosmology
    using CAMB, with the following parameters:
        Hubble constant (H0): 67.5 km/s/Mpc
        Baryon density (ombh2): 0.022
        Cold dark matter density (omch2): 0.122
        Neutrino mass sum (mnu): 0.06 eV
        Curvature (omk): 0
        Optical depth to reionization (tau): 0.04
        Scalar amplitude (As): 2e-9
        Scalar spectral index (ns): 0.965

    Returns
    -------
    l : ndarray
        Multipole moments (from 2 to 3000), dimensionless.
    EE : ndarray
        E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) in micro-Kelvin^2 (uK^2).
    """
    # Set cosmological parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0.0, tau=0.04)
    pars.InitPower.set_params(As=2e-9, ns=0.965)
    pars.set_for_lmax(3000, lens_potential_accuracy=0)
    pars.WantTensors = False

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000)
    cl = powers['total']  # shape: (lmax+1, 4): TT, EE, BB, TE

    # l values: 0 to lmax
    l = np.arange(cl.shape[0])
    # EE spectrum: cl[:,1] is C_l^{EE} in muK^2
    # Compute l(l+1)C_l^{EE}/(2pi)
    EE = np.zeros_like(l, dtype=np.float64)
    EE[2:] = l[2:] * (l[2:] + 1) * cl[2:, 1] / (2.0 * np.pi)  # units: muK^2

    # Restrict to l=2..3000
    l_out = l[2:3001]
    EE_out = EE[2:3001]

    return l_out, EE_out


def save_ee_spectrum_to_csv(l, EE, filename):
    r"""
    Save the E-mode polarization power spectrum to a CSV file.

    Parameters
    ----------
    l : ndarray
        Multipole moments (dimensionless).
    EE : ndarray
        E-mode polarization power spectrum (muK^2).
    filename : str
        Output CSV filename.
    """
    df = pd.DataFrame({'l': l, 'EE': EE})
    df.to_csv(filename, index=False)
    print("Saved E-mode polarization power spectrum to " + filename)
    print("First 5 rows:\n" + str(df.head()))
    print("Last 5 rows:\n" + str(df.tail()))


if __name__ == "__main__":
    l, EE = compute_cmb_ee_spectrum()
    save_ee_spectrum_to_csv(l, EE, "data/result.csv")