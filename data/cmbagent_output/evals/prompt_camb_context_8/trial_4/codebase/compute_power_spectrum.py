# filename: codebase/compute_power_spectrum.py
import camb
from camb import model
import numpy as np
import os

def compute_linear_matter_power_spectrum():
    r"""
    Compute the linear matter power spectrum P(k) at z=0 for a flat Lambda CDM cosmology
    using CAMB with specified parameters. The result is saved as a CSV file with columns:
    kh (h/Mpc), P_k ((Mpc/h)^3).

    Cosmological parameters:
        H0: Hubble constant [km/s/Mpc]
        ombh2: Baryon density Omega_b h^2 [dimensionless]
        omch2: Cold dark matter density Omega_c h^2 [dimensionless]
        mnu: Sum of neutrino masses [eV]
        omk: Curvature Omega_k [dimensionless]
        tau: Optical depth to reionization [dimensionless]
        As: Scalar amplitude [dimensionless]
        ns: Scalar spectral index [dimensionless]
        kmax: Maximum k for internal calculation [Mpc^-1]

    Output:
        CSV file with columns:
            kh: Wavenumber [h/Mpc]
            P_k: Linear matter power spectrum [(Mpc/h)^3]
    """
    # Output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cosmological parameters
    H0 = 67.5  # [km/s/Mpc]
    ombh2 = 0.022  # [dimensionless]
    omch2 = 0.122  # [dimensionless]
    mnu = 0.06  # [eV]
    omk = 0.0  # [dimensionless]
    tau = 0.06  # [dimensionless]
    As = 2.0e-9  # [dimensionless]
    ns = 0.965  # [dimensionless]
    kmax = 2.0  # [Mpc^-1]

    # Power spectrum output settings
    redshift = 0.0
    minkh = 1e-4  # [h/Mpc]
    maxkh = 1.0   # [h/Mpc]
    npoints = 200

    # Set up CAMB parameters
    pars = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu,
        omk=omk,
        tau=tau,
        As=As,
        ns=ns,
        WantTransfer=True
    )
    pars.NonLinear = model.NonLinear_none
    pars.set_matter_power(redshifts=[redshift], kmax=kmax)

    # Run CAMB
    results = camb.get_results(pars)

    # Get linear matter power spectrum
    kh, zs, pk = results.get_matter_power_spectrum(
        minkh=minkh,
        maxkh=maxkh,
        npoints=npoints
    )
    # pk shape: (len(zs), len(kh)), use first row (z=0)
    pk_z0 = pk[0, :]

    # Save to CSV
    output_file = os.path.join(output_dir, "result.csv")
    header = "kh (h/Mpc),P_k ((Mpc/h)^3)"
    data = np.column_stack((kh, pk_z0))
    np.savetxt(output_file, data, delimiter=",", header=header, comments='')

    # Print summary
    np.set_printoptions(precision=6, suppress=True)
    print("Linear matter power spectrum P(k) at z=0 computed with CAMB.")
    print("Cosmological parameters:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  Sum m_nu = " + str(mnu) + " eV")
    print("  Omega_k = " + str(omk))
    print("  tau = " + str(tau))
    print("  As = " + str(As))
    print("  ns = " + str(ns))
    print("  kmax = " + str(kmax) + " Mpc^-1")
    print("Output:")
    print("  File: " + output_file)
    print("  Number of k points: " + str(len(kh)))
    print("  kh range: " + str(kh[0]) + " to " + str(kh[-1]) + " h/Mpc")
    print("  P(k) range: " + str(pk_z0.min()) + " to " + str(pk_z0.max()) + " (Mpc/h)^3")

if __name__ == "__main__":
    compute_linear_matter_power_spectrum()
