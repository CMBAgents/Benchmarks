# filename: codebase/camb_power_spectrum.py
import camb
from camb import model
import numpy as np
import os

def compute_linear_matter_power_spectrum():
    r"""
    Computes the linear matter power spectrum P(k) at z=0 for a flat Lambda CDM cosmology
    using CAMB with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (Omega_b h^2): 0.022
        - Cold dark matter density (Omega_c h^2): 0.122
        - Neutrino mass sum (Sigma m_nu): 0.06 eV
        - Curvature (Omega_k): 0
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (A_s): 2e-9
        - Scalar spectral index (n_s): 0.965
        - k maximum (kmax): 2 (Mpc^-1)
    The function computes the linear matter power spectrum P(k) in units of (Mpc/h)^3
    for 200 logarithmically spaced k values in the range 1e-4 < kh < 1 (h/Mpc) at z=0,
    and saves the results in 'data/result.csv' with columns:
        - kh: Wavenumber (in h/Mpc)
        - P_k: Linear matter power spectrum (in (Mpc/h)^3)
    """
    # Ensure output directory exists
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
    As = 2e-9  # [dimensionless]
    ns = 0.965  # [dimensionless]
    kmax_calc = 2.0  # [Mpc^-1]

    # Output P(k) specifications
    z_output = 0.0  # [dimensionless]
    minkh_output = 1e-4  # [h/Mpc]
    maxkh_output = 1.0  # [h/Mpc]
    npoints_output = 200  # [dimensionless]

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=[z_output], kmax=kmax_calc)
    pars.NonLinear = model.NonLinear_none

    # Run CAMB and get results
    results = camb.get_results(pars)

    # Get the linear matter power spectrum
    kh_vals, z_vals, pk_vals = results.get_matter_power_spectrum(
        minkh=minkh_output, maxkh=maxkh_output, npoints=npoints_output
    )
    # pk_vals shape: (len(z_vals), len(kh_vals)), units: (Mpc/h)^3
    Pk_z0 = pk_vals[0, :]  # [ (Mpc/h)^3 ] at z=0

    # Save results to CSV
    output_data = np.column_stack((kh_vals, Pk_z0))
    output_file = os.path.join(output_dir, "result.csv")
    header_info = "kh (h/Mpc),P_k ((Mpc/h)^3)"
    np.savetxt(output_file, output_data, delimiter=",", header=header_info, comments='')

    # Print summary
    print("Linear matter power spectrum at z=0 computed and saved to data/result.csv")
    print("kh (h/Mpc) range: " + format(kh_vals[0], ".3e") + " to " + format(kh_vals[-1], ".3e"))
    print("P(k) ((Mpc/h)^3) range: " + format(Pk_z0[0], ".3e") + " to " + format(Pk_z0[-1], ".3e"))
    print("First 5 rows of the output (kh, P_k):")
    for i in range(5):
        print(format(kh_vals[i], ".5e") + ", " + format(Pk_z0[i], ".5e"))


if __name__ == "__main__":
    compute_linear_matter_power_spectrum()