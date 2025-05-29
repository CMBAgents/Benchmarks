# filename: codebase/compute_linear_matter_power_spectrum.py
import camb
from camb import model
import numpy as np
import os


def compute_linear_matter_power_spectrum():
    r"""Computes the linear matter power spectrum P(k) at redshift z=0 for a flat Lambda CDM cosmology
    using CAMB, with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (Omega_b h^2): 0.022
        - Cold dark matter density (Omega_c h^2): 0.122
        - Neutrino mass sum (Sigma m_nu): 0.06 eV
        - Curvature (Omega_k): 0
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (A_s): 2e-9
        - Scalar spectral index (n_s): 0.965
        - k maximum (kmax): 2 (Mpc^-1)
    The function computes P(k) in units of (Mpc/h)^3 for 200 logarithmically spaced k values
    in the range 1e-4 < k/h < 1 (h/Mpc) at z=0, and saves the results to 'data/result.csv'.
    """
    # Output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cosmological parameters
    H0 = 67.5  # [km/s/Mpc]
    ombh2 = 0.022  # Omega_b h^2
    omch2 = 0.122  # Omega_c h^2
    mnu = 0.06  # [eV]
    omk = 0.0  # Omega_k
    tau = 0.06  # Optical depth
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    kmax = 2.0  # [Mpc^-1] for internal CAMB calculation

    # Output P(k) specifications
    z_output = 0.0  # Redshift
    minkh = 1e-4  # [h/Mpc]
    maxkh = 1.0   # [h/Mpc]
    npoints = 200  # Number of k points (logarithmic spacing)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=[z_output], kmax=kmax)
    pars.NonLinear = model.NonLinear_none  # Linear power spectrum only

    # Run CAMB
    results = camb.get_results(pars)

    # Get linear matter power spectrum
    kh, zs, pk = results.get_matter_power_spectrum(
        minkh=minkh, maxkh=maxkh, npoints=npoints
    )
    # pk shape: (len(zs), len(kh)), units: (Mpc/h)^3
    Pk_z0 = pk[0, :]  # z=0

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    header = "kh (h/Mpc),P_k ((Mpc/h)^3)"
    data = np.column_stack((kh, Pk_z0))
    np.savetxt(output_path, data, delimiter=",", header=header, comments='')

    # Print summary
    print("Linear matter power spectrum P(k) at z=0 computed and saved to data/result.csv")
    print("Number of k points: " + str(npoints))
    print("k/h range: " + ("%.3e" % kh[0]) + " to " + ("%.3e" % kh[-1]) + " h/Mpc")
    print("P(k) range: " + ("%.3e" % Pk_z0[0]) + " to " + ("%.3e" % Pk_z0[-1]) + " (Mpc/h)^3")
    print("First 5 rows of the output (kh, P_k):")
    for i in range(5):
        print(("%.6e" % kh[i]) + ", " + ("%.6e" % Pk_z0[i]))


if __name__ == "__main__":
    compute_linear_matter_power_spectrum()