# filename: codebase/matter_power_spectrum.py
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower
import os

def compute_linear_matter_power_spectrum():
    r"""
    Computes the linear matter power spectrum P(k) at z=0 for specified cosmological parameters using CAMB.

    Returns:
        kh (np.ndarray): Wavenumber array in h/Mpc.
        pk (np.ndarray): Linear matter power spectrum in (Mpc/h)^3.
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density parameter [dimensionless]
    omch2 = 0.122  # Cold dark matter density parameter [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    As = 2e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]
    kmax = 2.0  # Maximum k [h/Mpc]
    num_k = 200  # Number of k points

    # k array in h/Mpc
    kh = np.linspace(1e-4, 1.0, num_k)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=[0.0], kmax=kmax)
    pars.NonLinear = model.NonLinear_none

    # Calculate results for these parameters
    results = camb.get_results(pars)
    # Get matter power spectrum interpolator
    PK = camb.get_matter_power_interpolator(pars, nonlinear=False, hubble_units=False, k_hunit=True, kmax=kmax)
    # Compute P(k) at z=0
    pk = PK.P(0.0, kh)  # P(k) in (Mpc/h)^3

    return kh, pk


def save_to_csv(kh, pk, filename):
    r"""
    Saves the wavenumber and power spectrum arrays to a CSV file.

    Args:
        kh (np.ndarray): Wavenumber array in h/Mpc.
        pk (np.ndarray): Power spectrum array in (Mpc/h)^3.
        filename (str): Output CSV filename.
    """
    df = pd.DataFrame({'kh': kh, 'P_k': pk})
    df.to_csv(filename, index=False)
    print("Saved linear matter power spectrum to " + filename)
    print("First 5 rows of the result:")
    print(df.head())
    print("Last 5 rows of the result:")
    print(df.tail())
    print("Total number of rows: " + str(len(df)))


if __name__ == "__main__":
    # Ensure data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    output_csv = os.path.join(data_dir, "result.csv")
    kh, pk = compute_linear_matter_power_spectrum()
    save_to_csv(kh, pk, output_csv)
