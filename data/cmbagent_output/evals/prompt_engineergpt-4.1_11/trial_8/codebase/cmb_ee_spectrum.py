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
    print("CAMB package is required but not installed. Please install camb and rerun the code.")
    raise e

def compute_cmb_ee_spectrum():
    """
    Computes the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for the specified cosmology.
    Returns:
        l_vals (np.ndarray): Multipole moments (l), shape (N,), unitless
        EE_vals (np.ndarray): E-mode power spectrum (micro-Kelvin^2), shape (N,), units: micro-Kelvin^2
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    tau = 0.1  # Optical depth to reionization [dimensionless]
    n_s = 0.95  # Scalar spectral index [dimensionless]
    # Scalar amplitude: 1.8e-9 * exp(2 * tau)
    A_s = 1.8e-9 * np.exp(2 * tau)  # [dimensionless]

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
    pars.InitPower.set_params(As=A_s, ns=n_s)
    pars.set_for_lmax(100, lens_potential_accuracy=0)
    # Exponential reionization with exponent power 2
    pars.Reion.use_optical_depth = True
    pars.Reion.reionization = True
    pars.Reion.reionization_scheme = camb.reionization.ReionizationScheme.Exp
    pars.Reion.exp_reion_exponent = 2.0

    # Get results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=100)
    # 'total' includes l=2 to l=100
    EE = powers['total'][:,2]  # EE spectrum, units: micro-Kelvin^2

    # l values: powers['total'] is indexed by l=0,1,...,lmax
    l_vals = np.arange(powers['total'].shape[0])
    # Select l=2 to l=100
    lmin = 2
    lmax = 100
    l_range = np.arange(lmin, lmax+1)
    # EE column: powers['total'][l, 2] is C_l^{EE} at l
    # Compute l(l+1)C_l^{EE}/(2pi)
    Cl_EE = powers['total'][l_range,2]  # units: micro-Kelvin^2
    EE_vals = l_range * (l_range + 1) * Cl_EE / (2.0 * np.pi)  # units: micro-Kelvin^2

    return l_range, EE_vals

def save_to_csv(l_vals, EE_vals, filename):
    """
    Save the multipole moments and E-mode power spectrum to a CSV file.
    Args:
        l_vals (np.ndarray): Multipole moments (l), shape (N,)
        EE_vals (np.ndarray): E-mode power spectrum (micro-Kelvin^2), shape (N,)
        filename (str): Output CSV filename
    """
    df = pd.DataFrame({'l': l_vals, 'EE': EE_vals})
    df.to_csv(filename, index=False)
    print("Saved E-mode polarization power spectrum to " + filename)
    # Print detailed results to console
    pd.set_option("display.max_rows", None)
    pd.set_option("display.precision", 6)
    print("\nE-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) [micro-Kelvin^2]:")
    print(df)
    pd.reset_option("display.max_rows")
    pd.reset_option("display.precision")

if __name__ == "__main__":
    l_vals, EE_vals = compute_cmb_ee_spectrum()
    save_to_csv(l_vals, EE_vals, "data/result.csv")
