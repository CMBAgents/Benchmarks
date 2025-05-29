# filename: codebase/compute_cmb_e_mode_spectrum.py
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
    print("CAMB is required for this script. Please install camb and rerun.")
    raise e


def compute_cmb_ee_spectrum():
    """
    Computes the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for the specified cosmology.

    Returns
    -------
    lvals : ndarray
        Multipole moments (l), shape (N,), unitless.
    EE : ndarray
        E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)), shape (N,), in micro-Kelvin^2.
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Omega_b h^2
    omch2 = 0.122  # Omega_c h^2
    tau = 0.1  # Optical depth to reionization
    n_s = 0.95  # Scalar spectral index
    # Scalar amplitude: A_s = 1.8e-9 * exp(2 * tau)
    A_s = 1.8e-9 * np.exp(2 * tau)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
    pars.InitPower.set_params(As=A_s, ns=n_s)
    pars.set_for_lmax(100, lens_potential_accuracy=0)

    # Set reionization model: Exponential with exponent power 2
    # (CAMB default is tanh, but supports exponential with exponent_power)
    pars.Reion.use_optical_depth = True
    pars.Reion.reionization = camb.reionization.ReionizationModel.exponential
    pars.Reion.exponent_power = 2.0

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=100)
    # 'total' includes l=0,1,2,...,lmax
    EE = powers['total'][:,2]  # EE spectrum, in muK^2

    # l values: powers['total'] is indexed from l=0
    lvals = np.arange(powers['total'].shape[0])

    # Compute l(l+1)C_l^{EE}/(2pi) for l=2 to l=100
    lmin = 2
    lmax = 100
    l = lvals[lmin:lmax+1]
    cl_ee = EE[lmin:lmax+1]
    ell_factor = l * (l + 1) / (2.0 * np.pi)
    EE_power = ell_factor * cl_ee  # in muK^2

    return l, EE_power


def save_to_csv(l, EE_power, filename):
    """
    Save the multipole moments and E-mode power spectrum to a CSV file.

    Parameters
    ----------
    l : ndarray
        Multipole moments (l), shape (N,), unitless.
    EE_power : ndarray
        E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)), shape (N,), in micro-Kelvin^2.
    filename : str
        Output CSV filename.
    """
    df = pd.DataFrame({'l': l, 'EE': EE_power})
    df.to_csv(filename, index=False)
    print("Saved E-mode polarization power spectrum to " + filename)
    print("First 10 rows of the result:")
    print(df.head(10).to_string(index=False))
    print("Last 5 rows of the result:")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    l, EE_power = compute_cmb_ee_spectrum()
    save_to_csv(l, EE_power, "data/result.csv")
