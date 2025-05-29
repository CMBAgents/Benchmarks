# filename: codebase/camb_ee_spectrum.py
import os
import numpy as np
import pandas as pd

# Ensure data directory exists
 data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

try:
    import camb
    from camb import model, initialpower
except ImportError as e:
    print("CAMB package is required but not installed. Please install camb and rerun the code.")
    raise e


def compute_cmb_ee_spectrum():
    """
    Computes the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for a flat Lambda CDM cosmology
    using CAMB, with specified cosmological parameters and exponential reionization model (exponent power 2).
    Returns:
        l_vals (np.ndarray): Multipole moments (l), shape (N,), unitless
        EE_spectrum (np.ndarray): E-mode power spectrum (l(l+1)C_l^{EE}/(2pi)), shape (N,), units: micro-Kelvin^2 (uK^2)
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant, units: km/s/Mpc
    ombh2 = 0.022  # Baryon density, units: dimensionless (Omega_b h^2)
    omch2 = 0.122  # Cold dark matter density, units: dimensionless (Omega_c h^2)
    tau = 0.1  # Optical depth to reionization, unitless
    n_s = 0.95  # Scalar spectral index, unitless
    # Scalar amplitude, units: dimensionless
    # A_s = 1.8e-9 * exp(2 * tau)
    A_s = 1.8e-9 * np.exp(2 * tau)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0, tau=tau)
    pars.InitPower.set_params(As=A_s, ns=n_s, r=0)
    # Exponential reionization model with exponent power 2
    pars.Reion.use_optical_depth = True
    pars.Reion.reionization = True
    pars.Reion.reionization_scheme = camb.reionization.ReionizationScheme.Exp
    pars.Reion.exp_reion_exponent = 2.0

    # Set lmax to at least 100 for required range
    lmax = 100
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total'])
    totCL = powers['total']  # Shape: (lmax+1, 4): TT, EE, BB, TE

    # l values: from 0 to lmax
    l_vals_full = np.arange(totCL.shape[0])
    # EE spectrum: column 1, units: muK^2
    EE_full = totCL[:, 1]

    # Compute l(l+1)C_l^{EE}/(2pi) for l=2 to l=100
    l_min = 2
    l_max = 100
    l_vals = l_vals_full[l_min:l_max+1]
    EE_spectrum = (l_vals * (l_vals + 1) * EE_full[l_min:l_max+1]) / (2.0 * np.pi)  # units: muK^2

    return l_vals, EE_spectrum


def save_ee_spectrum_to_csv(l_vals, EE_spectrum, filename):
    """
    Saves the E-mode power spectrum to a CSV file.
    Args:
        l_vals (np.ndarray): Multipole moments (l), shape (N,)
        EE_spectrum (np.ndarray): E-mode power spectrum, shape (N,), units: micro-Kelvin^2 (uK^2)
        filename (str): Output CSV filename
    """
    df = pd.DataFrame({'l': l_vals.astype(int), 'EE': EE_spectrum})
    df.to_csv(filename, index=False)
    print("Saved E-mode polarization power spectrum to " + filename)
    print("First 10 rows of the result:")
    print(df.head(10).to_string(index=False))
    print("\nLast 10 rows of the result:")
    print(df.tail(10).to_string(index=False))


if __name__ == "__main__":
    l_vals, EE_spectrum = compute_cmb_ee_spectrum()
    output_csv = os.path.join("data", "result.csv")
    save_ee_spectrum_to_csv(l_vals, EE_spectrum, output_csv)
