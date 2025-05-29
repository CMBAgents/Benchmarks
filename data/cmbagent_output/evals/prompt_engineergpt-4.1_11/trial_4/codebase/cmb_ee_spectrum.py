# filename: codebase/cmb_ee_spectrum.py
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
    """
    Computes the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for the specified cosmology.
    Returns:
        l_vals (np.ndarray): Multipole moments (l), shape (N,), unitless (integer)
        EE_vals (np.ndarray): E-mode power spectrum (micro-Kelvin^2), shape (N,)
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [Omega_b h^2]
    omch2 = 0.122  # Cold dark matter density [Omega_c h^2]
    tau = 0.1  # Optical depth to reionization [dimensionless]
    n_s = 0.95  # Scalar spectral index [dimensionless]
    # Scalar amplitude: 1.8e-9 * exp(2 * tau)
    A_s = 1.8e-9 * np.exp(2 * tau)  # [dimensionless]

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
    pars.InitPower.set_params(As=A_s, ns=n_s)
    # Exponential reionization with exponent power 2
    pars.Reion.use_optical_depth = True
    pars.Reion.reionization = camb.reionization.Reionization()
    pars.Reion.reionization.reionization = camb.reionization.ReionizationModel.exp
    pars.Reion.reionization.exponent = 2.0

    # Set lmax to at least 100 for required range
    pars.set_for_lmax(100, lens_potential_accuracy=0)
    # Request the EE spectrum
    pars.WantTensors = False
    pars.Want_CMB = True

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=100)
    # 'total' includes l=2 to l=100
    EE = powers['total'][:, 1]  # EE spectrum in muK^2

    # l values: powers['total'] is indexed from l=0, so l=2 is index 2
    l_vals = np.arange(powers['total'].shape[0])
    lmin = 2
    lmax = 100
    l_range = np.arange(lmin, lmax + 1)
    EE_range = EE[lmin:lmax + 1]

    # Compute l(l+1)C_l^{EE}/(2pi)
    factor = l_range * (l_range + 1) / (2.0 * np.pi)
    EE_power = factor * EE_range  # [muK^2]

    return l_range, EE_power

def save_to_csv(l_vals, EE_vals, filename):
    """
    Save the multipole moments and E-mode power spectrum to a CSV file.
    Args:
        l_vals (np.ndarray): Multipole moments (l), shape (N,)
        EE_vals (np.ndarray): E-mode power spectrum (micro-Kelvin^2), shape (N,)
        filename (str): Output CSV file path
    """
    df = pd.DataFrame({'l': l_vals, 'EE': EE_vals})
    df.to_csv(filename, index=False)
    print("Saved E-mode power spectrum to " + filename)
    # Print detailed results to console
    pd.set_option("display.max_rows", None)
    pd.set_option("display.precision", 6)
    print("\nE-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) [muK^2]:")
    print(df)
    pd.reset_option("display.max_rows")
    pd.reset_option("display.precision")

if __name__ == "__main__":
    l_vals, EE_vals = compute_cmb_ee_spectrum()
    save_to_csv(l_vals, EE_vals, "data/result.csv")
