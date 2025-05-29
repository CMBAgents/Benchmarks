# filename: codebase/power_spectrum.py
import numpy as np
import pandas as pd
import os
import camb
from camb import model, initialpower

def compute_linear_matter_power_spectrum():
    r"""
    Computes the linear matter power spectrum P(k) at z=0 for a flat Lambda CDM cosmology using CAMB.

    Returns
    -------
    kh : ndarray
        Wavenumber array in h/Mpc (shape: (200,))
    pk : ndarray
        Linear matter power spectrum in (Mpc/h)^3 (shape: (200,))
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
    z = 0.0  # Redshift

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=[z], kmax=kmax)
    pars.NonLinear = model.NonLinear_none

    # Generate k array (in h/Mpc)
    kh = np.linspace(1e-4, 1.0, 200)  # 200 points from 1e-4 to 1 (h/Mpc)

    # Calculate results for these parameters
    results = camb.get_results(pars)
    # Get matter power spectrum interpolator
    PK = camb.get_matter_power_interpolator(pars, nonlinear=False, hubble_units=False, k_hunit=True, kmax=kmax, zmax=0.0)
    # Compute P(k) at z=0
    pk = PK.P(z, kh)  # pk in (Mpc/h)^3

    return kh, pk

def save_results_to_csv(kh, pk, filename):
    r"""
    Save the computed kh and P_k arrays to a CSV file.

    Parameters
    ----------
    kh : ndarray
        Wavenumber array in h/Mpc.
    pk : ndarray
        Linear matter power spectrum in (Mpc/h)^3.
    filename : str
        Output CSV filename.
    """
    df = pd.DataFrame({'kh': kh, 'P_k': pk})
    df.to_csv(filename, index=False)

def print_summary(kh, pk):
    r"""
    Print a detailed summary of the computed power spectrum.

    Parameters
    ----------
    kh : ndarray
        Wavenumber array in h/Mpc.
    pk : ndarray
        Linear matter power spectrum in (Mpc/h)^3.
    """
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print("\nLinear matter power spectrum P(k) at z=0 for flat Lambda CDM cosmology:")
    print("Number of k points: " + str(len(kh)))
    print("k range: " + ("%.4e" % kh[0]) + " to " + ("%.4e" % kh[-1]) + " h/Mpc")
    print("P(k) range: " + ("%.4e" % np.min(pk)) + " to " + ("%.4e" % np.max(pk)) + " (Mpc/h)^3")
    print("\nFirst 5 rows:")
    for i in range(5):
        print("k = " + ("%.6e" % kh[i]) + " h/Mpc, P(k) = " + ("%.6e" % pk[i]) + " (Mpc/h)^3")
    print("\nLast 5 rows:")
    for i in range(-5, 0):
        print("k = " + ("%.6e" % kh[i]) + " h/Mpc, P(k) = " + ("%.6e" % pk[i]) + " (Mpc/h)^3")

if __name__ == "__main__":
    # Ensure data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    output_csv = os.path.join(data_dir, "result.csv")

    # Compute power spectrum
    kh, pk = compute_linear_matter_power_spectrum()
    # Save to CSV
    save_results_to_csv(kh, pk, output_csv)
    print("Results saved to " + output_csv)
    # Print summary
    print_summary(kh, pk)