# filename: codebase/camb_power_spectrum.py
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower
import os

def compute_linear_matter_power_spectrum(
    H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0.0, tau=0.06,
    As=2e-9, ns=0.965, kmin=1e-4, kmax=1.0, num_k=200, z=0.0, output_csv='data/result.csv'
):
    r"""
    Compute the linear matter power spectrum P(k) at redshift z for a flat Lambda CDM cosmology.

    Parameters
    ----------
    H0 : float
        Hubble constant [km/s/Mpc]
    ombh2 : float
        Baryon density parameter [dimensionless]
    omch2 : float
        Cold dark matter density parameter [dimensionless]
    mnu : float
        Sum of neutrino masses [eV]
    omk : float
        Curvature parameter [dimensionless]
    tau : float
        Optical depth to reionization [dimensionless]
    As : float
        Scalar amplitude [dimensionless]
    ns : float
        Scalar spectral index [dimensionless]
    kmin : float
        Minimum wavenumber [h/Mpc]
    kmax : float
        Maximum wavenumber [h/Mpc]
    num_k : int
        Number of k points [dimensionless]
    z : float
        Redshift [dimensionless]
    output_csv : str
        Output CSV file path

    Returns
    -------
    None
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=[z], kmax=2.0)
    pars.NonLinear = model.NonLinear_none

    # Calculate results for these parameters
    results = camb.get_results(pars)
    kh = np.linspace(kmin, kmax, num_k)  # [h/Mpc]
    # Get matter power spectrum interpolator
    PK = camb.get_matter_power_interpolator(pars, nonlinear=False, hubble_units=False, k_hunit=True, kmax=2.0, zmax=0.0)
    P_k = PK.P(z, kh)  # [Mpc/h]^3

    # Save to CSV
    df = pd.DataFrame({'kh': kh, 'P_k': P_k})
    df.to_csv(output_csv, index=False)

    # Print summary
    np.set_printoptions(precision=6, suppress=True)
    print("Linear matter power spectrum P(k) at z=0 computed for 200 k values in the range 1e-4 < kh < 1 (h/Mpc).")
    print("Cosmological parameters used:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  Sum m_nu = " + str(mnu) + " eV")
    print("  Omega_k = " + str(omk))
    print("  tau = " + str(tau))
    print("  A_s = " + str(As))
    print("  n_s = " + str(ns))
    print("  k range: " + str(kmin) + " to " + str(kmax) + " h/Mpc")
    print("First 5 rows of the result:")
    print(df.head())
    print("Results saved to " + output_csv)


if __name__ == "__main__":
    compute_linear_matter_power_spectrum()