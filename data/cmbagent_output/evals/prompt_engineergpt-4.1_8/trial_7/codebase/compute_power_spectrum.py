# filename: codebase/compute_power_spectrum.py
import numpy as np
import pandas as pd
import os
import camb
from camb import model, initialpower

def compute_linear_matter_power_spectrum():
    r"""
    Compute the linear matter power spectrum P(k) at z=0 for a flat Lambda CDM cosmology
    using CAMB with the following parameters:
        Hubble constant (H0): 67.5 km/s/Mpc
        Baryon density (Omega_b h^2): 0.022
        Cold dark matter density (Omega_c h^2): 0.122
        Neutrino mass sum (sum_mnu): 0.06 eV
        Curvature (Omega_k): 0
        Optical depth to reionization (tau): 0.06
        Scalar amplitude (A_s): 2e-9
        Scalar spectral index (n_s): 0.965
        k maximum (kmax): 2 (Mpc^-1)
    The function computes P(k) in units of (Mpc/h)^3 for 200 evenly spaced k values in the range
    1e-4 < kh < 1 (Mpc^-1), and saves the results in 'data/result.csv' with columns:
        kh: Wavenumber (in h/Mpc)
        P_k: Linear matter power spectrum (in (Mpc/h)^3)
    """
    # Output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "result.csv")

    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Omega_b h^2
    omch2 = 0.122  # Omega_c h^2
    mnu = 0.06  # sum of neutrino masses [eV]
    omk = 0.0  # curvature
    tau = 0.06  # optical depth
    As = 2e-9  # scalar amplitude
    ns = 0.965  # scalar spectral index
    kmax = 2.0  # maximum k [Mpc^-1]

    # k values (in h/Mpc)
    kh_vals = np.linspace(1e-4, 1.0, 200)  # 200 points from 1e-4 to 1

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=[0.0], kmax=kmax)
    pars.NonLinear = model.NonLinear_none  # Linear power spectrum only

    # Calculate results
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=kh_vals[0], maxkh=kh_vals[-1], npoints=len(kh_vals))
    # kh: array of k values in h/Mpc
    # pk: shape (len(z), len(kh)), units (Mpc/h)^3

    # Interpolate pk to the requested kh_vals if necessary
    # Since get_matter_power_spectrum returns kh and pk arrays, but kh may not match kh_vals exactly
    # We'll interpolate pk[0] (z=0) to kh_vals
    from scipy.interpolate import interp1d
    pk_interp_func = interp1d(kh, pk[0], kind='linear', bounds_error=False, fill_value='extrapolate')
    pk_vals = pk_interp_func(kh_vals)

    # Save to CSV
    df = pd.DataFrame({'kh': kh_vals, 'P_k': pk_vals})
    df.to_csv(output_file, index=False)

    # Print summary
    np.set_printoptions(precision=6, suppress=True)
    print("Linear matter power spectrum P(k) at z=0 computed for 200 k values in the range 1e-4 < kh < 1 (h/Mpc).")
    print("Cosmological parameters used:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  sum m_nu = " + str(mnu) + " eV")
    print("  Omega_k = " + str(omk))
    print("  tau = " + str(tau))
    print("  A_s = " + str(As))
    print("  n_s = " + str(ns))
    print("  kmax = " + str(kmax) + " Mpc^-1")
    print("Results saved to: " + output_file)
    print("First 5 rows of the result:")
    print(df.head())
    print("Last 5 rows of the result:")
    print(df.tail())


if __name__ == "__main__":
    compute_linear_matter_power_spectrum()