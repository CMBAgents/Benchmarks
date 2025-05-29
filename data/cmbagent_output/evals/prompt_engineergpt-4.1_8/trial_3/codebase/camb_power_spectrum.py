# filename: codebase/camb_power_spectrum.py
import numpy as np
import pandas as pd
import os

import camb
from camb import model, initialpower

def compute_linear_matter_power_spectrum():
    r"""
    Compute the linear matter power spectrum P(k) at z=0 for specified cosmological parameters using CAMB.

    Returns
    -------
    kh : ndarray
        Wavenumber array in h/Mpc.
    pk : ndarray
        Linear matter power spectrum in (Mpc/h)^3.
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Omega_b h^2
    omch2 = 0.122  # Omega_c h^2
    mnu = 0.06  # sum m_nu [eV]
    omk = 0.0  # Omega_k
    tau = 0.06  # optical depth
    As = 2e-9  # scalar amplitude
    ns = 0.965  # scalar spectral index
    kmax = 2.0  # maximum k [h/Mpc]

    # k range (in h/Mpc)
    kmin = 1e-4
    kmax_plot = 1.0
    num_k = 200
    kh = np.linspace(kmin, kmax_plot, num_k)  # h/Mpc

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=[0.0], kmax=kmax)
    pars.NonLinear = model.NonLinear_none

    # Calculate results for these parameters
    results = camb.get_results(pars)
    kh_camb, z_camb, pk_camb = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints=500)
    # pk_camb shape: (len(z_camb), len(kh_camb))

    # Interpolate P(k) at z=0 for the requested kh values
    pk_z0 = np.interp(kh, kh_camb, pk_camb[0])  # (Mpc/h)^3

    return kh, pk_z0

def save_to_csv(kh, pk, filename):
    r"""
    Save the wavenumber and power spectrum arrays to a CSV file.

    Parameters
    ----------
    kh : ndarray
        Wavenumber array in h/Mpc.
    pk : ndarray
        Power spectrum array in (Mpc/h)^3.
    filename : str
        Output CSV filename.
    """
    df = pd.DataFrame({'kh': kh, 'P_k': pk})
    df.to_csv(filename, index=False)

def print_summary(kh, pk):
    r"""
    Print a summary of the computed power spectrum.

    Parameters
    ----------
    kh : ndarray
        Wavenumber array in h/Mpc.
    pk : ndarray
        Power spectrum array in (Mpc/h)^3.
    """
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print("\nSummary of computed linear matter power spectrum at z=0:")
    print("Number of k points: " + str(len(kh)))
    print("k range: %.2e to %.2e h/Mpc" % (kh[0], kh[-1]))
    print("P(k) range: %.2e to %.2e (Mpc/h)^3" % (pk.min(), pk.max()))
    print("\nFirst 5 rows:")
    for i in range(5):
        print("k = %.6e h/Mpc, P(k) = %.6e (Mpc/h)^3" % (kh[i], pk[i]))
    print("\nLast 5 rows:")
    for i in range(-5, 0):
        print("k = %.6e h/Mpc, P(k) = %.6e (Mpc/h)^3" % (kh[i], pk[i]))

if __name__ == "__main__":
    # Ensure data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Compute power spectrum
    kh, pk = compute_linear_matter_power_spectrum()

    # Save to CSV
    output_csv = os.path.join(data_dir, "result.csv")
    save_to_csv(kh, pk, output_csv)

    # Print summary
    print_summary(kh, pk)
    print("\nResults saved to " + output_csv)