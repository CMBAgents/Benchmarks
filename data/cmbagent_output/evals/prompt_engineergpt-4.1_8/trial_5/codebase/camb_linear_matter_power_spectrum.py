# filename: codebase/camb_linear_matter_power_spectrum.py
import numpy as np
import pandas as pd
import os
import camb
from camb import model, initialpower

def compute_linear_matter_power_spectrum():
    r"""
    Computes the linear matter power spectrum P(k) at z=0 for a flat Lambda CDM cosmology
    using CAMB with specified parameters. Saves the results to 'data/result.csv'.

    Returns:
        None
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

    # k range for output (in h/Mpc)
    kmin = 1e-4
    kmax_out = 1.0
    num_k = 200
    kh = np.linspace(kmin, kmax_out, num_k)  # [h/Mpc]

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=[0.0], kmax=kmax)
    pars.NonLinear = model.NonLinear_none  # Linear power spectrum

    # Calculate results
    results = camb.get_results(pars)
    kh_camb, z_camb, pk_camb = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints=500)
    # pk_camb shape: (len(z_camb), len(kh_camb)), units: (Mpc/h)^3

    # Interpolate P(k) at z=0 for requested kh values
    pk_z0 = np.interp(kh, kh_camb, pk_camb[0])  # [Mpc/h]^3

    # Prepare output DataFrame
    df = pd.DataFrame({'kh': kh, 'P_k': pk_z0})

    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to CSV
    output_file = os.path.join(output_dir, "result.csv")
    df.to_csv(output_file, index=False)

    # Print summary to console
    np.set_printoptions(precision=6, suppress=True)
    print("Linear matter power spectrum P(k) at z=0 computed for 200 k values.")
    print("Cosmological parameters:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  ombh2 = " + str(ombh2))
    print("  omch2 = " + str(omch2))
    print("  mnu = " + str(mnu) + " eV")
    print("  omk = " + str(omk))
    print("  tau = " + str(tau))
    print("  As = " + str(As))
    print("  ns = " + str(ns))
    print("k range: " + str(kmin) + " < kh < " + str(kmax_out) + " [h/Mpc], number of points: " + str(num_k))
    print("Results saved to: " + output_file)
    print("First 5 rows of the result:")
    print(df.head())
    print("Last 5 rows of the result:")
    print(df.tail())

if __name__ == "__main__":
    compute_linear_matter_power_spectrum()