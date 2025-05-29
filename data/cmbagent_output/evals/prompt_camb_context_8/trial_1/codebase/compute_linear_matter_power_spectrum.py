# filename: codebase/compute_linear_matter_power_spectrum.py
r"""
Compute the linear matter power spectrum P(k) at z=0 for a flat Lambda CDM cosmology using CAMB,
with the following parameters:
    H0 = 67.5 km/s/Mpc
    Omega_b h^2 = 0.022
    Omega_c h^2 = 0.122
    Sum m_nu = 0.06 eV
    Omega_k = 0
    tau = 0.06
    A_s = 2e-9
    n_s = 0.965
    kmax = 2.0 (Mpc^-1)
Output:
    - 200 logarithmically spaced k/h values in [1e-4, 1] (h/Mpc)
    - Linear matter power spectrum P(k) at z=0 in (Mpc/h)^3
    - Save as data/result.csv with columns: kh, P_k
"""

import camb
from camb import model
import numpy as np
import os


def compute_linear_matter_power_spectrum():
    r"""
    Computes the linear matter power spectrum P(k) at z=0 for a flat Lambda CDM cosmology
    using CAMB, with specified cosmological parameters. The result is saved to data/result.csv
    with columns:
        kh: Wavenumber in h/Mpc (logarithmically spaced, 200 points, 1e-4 to 1)
        P_k: Linear matter power spectrum in (Mpc/h)^3 at z=0

    Returns
    -------
    None
    """
    # Cosmological parameters (units in comments)
    H0 = 67.5            # Hubble constant [km/s/Mpc]
    ombh2 = 0.022        # Omega_b h^2 [dimensionless]
    omch2 = 0.122        # Omega_c h^2 [dimensionless]
    mnu = 0.06           # Sum of neutrino masses [eV]
    omk = 0.0            # Curvature Omega_k [dimensionless]
    tau = 0.06           # Optical depth to reionization [dimensionless]
    As = 2e-9            # Scalar amplitude [dimensionless]
    ns = 0.965           # Scalar spectral index [dimensionless]
    kmax = 2.0           # Maximum k for calculation [Mpc^-1]

    # Output P(k) specifications
    z_output = 0.0
    minkh = 1e-4         # Minimum k/h [h/Mpc]
    maxkh = 1.0          # Maximum k/h [h/Mpc]
    npoints = 200        # Number of k/h points (logarithmically spaced)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=[z_output], kmax=kmax)
    pars.NonLinear = model.NonLinear_none

    # Run CAMB
    results = camb.get_results(pars)

    # Get linear matter power spectrum at z=0
    kh, zs, pk = results.get_matter_power_spectrum(minkh=minkh, maxkh=maxkh, npoints=npoints)
    # pk shape: (len(zs), len(kh)), units: (Mpc/h)^3
    Pk_z0 = pk[0, :]  # z=0

    # Prepare output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to CSV
    output_file = os.path.join(output_dir, "result.csv")
    header = "kh (h/Mpc),P_k ((Mpc/h)^3)"
    data = np.column_stack((kh, Pk_z0))
    np.savetxt(output_file, data, delimiter=",", header=header, comments='')

    # Print summary
    print("Linear matter power spectrum P(k) at z=0 computed and saved to data/result.csv")
    print("kh (h/Mpc) range: " + ("%.3e" % kh[0]) + " to " + ("%.3e" % kh[-1]))
    print("P_k ((Mpc/h)^3) range: " + ("%.3e" % Pk_z0[0]) + " to " + ("%.3e" % Pk_z0[-1]))
    print("First 5 rows (kh, P_k):")
    for i in range(5):
        print(("%.5e" % kh[i]) + ", " + ("%.5e" % Pk_z0[i]))


if __name__ == "__main__":
    compute_linear_matter_power_spectrum()
