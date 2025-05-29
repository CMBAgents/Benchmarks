# filename: codebase/linear_matter_power_spectrum.py
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower
import os

def compute_linear_matter_power_spectrum():
    r"""
    Computes the linear matter power spectrum P(k) at z=0 for specified cosmological parameters using CAMB.

    Returns:
        kh (np.ndarray): Wavenumbers in h/Mpc.
        pk (np.ndarray): Linear matter power spectrum in (Mpc/h)^3.
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density parameter
    omch2 = 0.122  # Cold dark matter density parameter
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    kmax = 2.0  # Maximum k [h/Mpc]

    # k range
    kmin = 1e-4  # Minimum k [h/Mpc]
    kmax_plot = 1.0  # Maximum k for output [h/Mpc]
    num_k = 200  # Number of k points

    kh = np.linspace(kmin, kmax_plot, num_k)  # [h/Mpc]

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=[0.0], kmax=kmax)
    pars.NonLinear = model.NonLinear_none  # Linear power spectrum

    # Calculate results
    results = camb.get_results(pars)
    kh_camb, z_camb, pk_camb = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints=500)
    # pk_camb shape: (len(z_camb), len(kh_camb))

    # Interpolate to requested kh
    pk_interp = np.interp(kh, kh_camb, pk_camb[0])  # z=0

    return kh, pk_interp

def save_to_csv(kh, pk, filename):
    r"""
    Saves the wavenumber and power spectrum arrays to a CSV file.

    Args:
        kh (np.ndarray): Wavenumbers in h/Mpc.
        pk (np.ndarray): Power spectrum in (Mpc/h)^3.
        filename (str): Output CSV filename.
    """
    df = pd.DataFrame({'kh': kh, 'P_k': pk})
    df.to_csv(filename, index=False)

def print_summary(kh, pk):
    r"""
    Prints a summary of the computed power spectrum to the console.

    Args:
        kh (np.ndarray): Wavenumbers in h/Mpc.
        pk (np.ndarray): Power spectrum in (Mpc/h)^3.
    """
    np.set_printoptions(precision=6, suppress=True)
    print("Linear matter power spectrum P(k) at z=0 (units: kh [h/Mpc], P_k [(Mpc/h)^3]):")
    print("Number of k points: " + str(len(kh)))
    print("k range: {:.4e} to {:.4e} h/Mpc".replace("{:.4e}", "%.4e") % (kh[0], kh[-1]))
    print("P(k) range: {:.4e} to {:.4e} (Mpc/h)^3".replace("{:.4e}", "%.4e") % (pk.min(), pk.max()))
    print("First 5 rows:")
    for i in range(min(5, len(kh))):
        print("kh = %.6e h/Mpc, P_k = %.6e (Mpc/h)^3" % (kh[i], pk[i]))
    print("Last 5 rows:")
    for i in range(-5, 0):
        print("kh = %.6e h/Mpc, P_k = %.6e (Mpc/h)^3" % (kh[i], pk[i]))

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
