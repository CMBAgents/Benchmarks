# filename: codebase/camb_ee_spectrum.py
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


def compute_cmb_ee_spectrum():
    """
    Computes the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for a flat Lambda CDM cosmology
    using CAMB, with specified cosmological parameters and reionization model.

    Returns
    -------
    lvals : ndarray
        Multipole moments (l), shape (N,), unitless (integer)
    EE : ndarray
        E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)), shape (N,), units: micro-Kelvin^2
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    tau = 0.1  # Optical depth to reionization [dimensionless]
    n_s = 0.95  # Scalar spectral index [dimensionless]
    # Scalar amplitude [dimensionless], A_s = 1.8e-9 * exp(2 * tau)
    A_s = 1.8e-9 * np.exp(2 * tau)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0, tau=tau)
    pars.InitPower.set_params(As=A_s, ns=n_s, r=0)
    # Exponential reionization with exponent power 2
    pars.Reion.use_optical_depth = True
    pars.Reion.reionization = True
    pars.Reion.reionization_scheme = camb.reionization.ReionizationScheme.Exp
    pars.Reion.exp_reion_exponent = 2.0

    # Set lmax to at least 100 for required range
    pars.set_for_lmax(100, lens_potential_accuracy=0)

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=100)
    # 'total' includes l=2 to l=100
    cl = powers['total']  # shape: (lmax+1, 4), columns: TT, EE, BB, TE

    # l values: cl[2] corresponds to l=2, etc.
    lvals = np.arange(2, 101)  # l=2 to l=100
    # EE power spectrum: cl[:,1] is C_l^{EE} in muK^2
    cl_ee = cl[2:101, 1]  # units: muK^2

    # Compute l(l+1)C_l^{EE}/(2pi) [muK^2]
    factor = lvals * (lvals + 1) / (2.0 * np.pi)
    EE = factor * cl_ee  # units: muK^2

    return lvals, EE


def save_to_csv(lvals, EE, filename):
    """
    Save the multipole moments and E-mode power spectrum to a CSV file.

    Parameters
    ----------
    lvals : ndarray
        Multipole moments (l), shape (N,), unitless (integer)
    EE : ndarray
        E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)), shape (N,), units: micro-Kelvin^2
    filename : str
        Output CSV filename
    """
    df = pd.DataFrame({'l': lvals, 'EE': EE})
    df.to_csv(filename, index=False)
    print("Saved E-mode polarization power spectrum to " + filename)


def print_results(lvals, EE):
    """
    Print the E-mode power spectrum results to the console in a detailed manner.

    Parameters
    ----------
    lvals : ndarray
        Multipole moments (l), shape (N,), unitless (integer)
    EE : ndarray
        E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)), shape (N,), units: micro-Kelvin^2
    """
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print("CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) [muK^2] for l=2 to l=100:")
    print("l\tEE [muK^2]")
    for l, ee in zip(lvals, EE):
        print(str(int(l)) + "\t" + str(ee))


if __name__ == "__main__":
    lvals, EE = compute_cmb_ee_spectrum()
    save_to_csv(lvals, EE, "data/result.csv")
    print_results(lvals, EE)
