# filename: codebase/compute_power_spectrum.py
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower
import os

def compute_linear_matter_power_spectrum():
    r"""
    Compute the linear matter power spectrum P(k) at z=0 for a flat Lambda CDM cosmology using CAMB.

    Cosmological parameters:
        Hubble constant (H0): 67.5 km/s/Mpc
        Baryon density (ombh2): 0.022
        Cold dark matter density (omch2): 0.122
        Neutrino mass sum (mnu): 0.06 eV
        Curvature (omk): 0
        Optical depth to reionization (tau): 0.06
        Scalar amplitude (As): 2e-9
        Scalar spectral index (ns): 0.965
        k maximum (kmax): 2

    Computes P(k) in units of (Mpc/h)^3 for 200 evenly spaced k values in the range 1e-4 < kh < 1 (h/Mpc).
    Saves the results in 'data/result.csv' with columns:
        kh: Wavenumber (in h/Mpc)
        P_k: Linear matter power spectrum (in (Mpc/h)^3)
    """
    # Ensure data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Cosmological parameters
    H0 = 67.5  # km/s/Mpc
    ombh2 = 0.022  # Omega_b h^2
    omch2 = 0.122  # Omega_c h^2
    mnu = 0.06  # eV
    omk = 0.0  # Flat universe
    tau = 0.06
    As = 2e-9
    ns = 0.965
    kmax = 2.0  # Maximum k in h/Mpc

    # k values (in h/Mpc)
    kmin = 1e-4
    kmax_plot = 1.0
    num_k = 200
    kh = np.linspace(kmin, kmax_plot, num_k)  # h/Mpc

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=[0.0], kmax=kmax)
    pars.NonLinear = model.NonLinear_none  # Linear power spectrum

    # Calculate results
    results = camb.get_results(pars)
    kh_camb, z_camb, pk_camb = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints=1000)
    # pk_camb shape: (len(z_camb), len(kh_camb))

    # Interpolate P(k) at z=0 for requested kh values
    pk_z0 = np.interp(kh, kh_camb, pk_camb[0])  # (Mpc/h)^3

    # Save to CSV
    df = pd.DataFrame({'kh': kh, 'P_k': pk_z0})
    output_file = os.path.join(data_dir, "result.csv")
    df.to_csv(output_file, index=False)

    # Print summary
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print("Linear matter power spectrum P(k) at z=0 computed for 200 k values in the range 1e-4 < kh < 1 (h/Mpc).")
    print("Cosmological parameters:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  Neutrino mass sum = " + str(mnu) + " eV")
    print("  Omega_k = " + str(omk))
    print("  tau = " + str(tau))
    print("  As = " + str(As))
    print("  ns = " + str(ns))
    print("  kmax = " + str(kmax) + " h/Mpc")
    print("First 5 rows of the result (kh in h/Mpc, P_k in (Mpc/h)^3):")
    print(df.head())
    print("Results saved to " + output_file)


if __name__ == "__main__":
    compute_linear_matter_power_spectrum()