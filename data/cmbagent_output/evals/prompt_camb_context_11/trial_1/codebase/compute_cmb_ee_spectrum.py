# filename: codebase/compute_cmb_ee_spectrum.py
import camb
import numpy as np
import math
import os

def compute_cmb_ee_spectrum():
    r"""
    Compute the CMB E-mode polarization power spectrum (Dl^EE = l(l+1)Cl^EE/2pi) for a flat Lambda CDM cosmology
    using CAMB with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (Omega_b h^2): 0.022
        - Cold dark matter density (Omega_c h^2): 0.122
        - Scalar amplitude (A_s): 1.8e-9 * exp(2 * tau)
        - Scalar spectral index (n_s): 0.95
        - Optical depth to reionization (tau): 0.1
        - Reionization model: Exponential reionization with exponent power 2

    The E-mode power spectrum (Dl^EE) is computed in units of microKelvin^2 for multipole moments l=2 to l=100.
    The results are saved in 'data/result.csv' with columns:
        l: Multipole moment (integer values from 2 to 100)
        EE: E-mode polarization power spectrum (Dl^EE) in microKelvin^2

    Returns
    -------
    None
    """
    # Cosmological parameters (units in comments)
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density Omega_b * h^2 [dimensionless]
    omch2 = 0.122  # Cold dark matter density Omega_c * h^2 [dimensionless]
    tau = 0.1  # Optical depth to reionization [dimensionless]
    ns = 0.95  # Scalar spectral index [dimensionless]
    omk = 0.0  # Curvature density Omega_k [dimensionless, 0 for flat]
    As = 1.8e-9 * math.exp(2 * tau)  # Scalar amplitude [dimensionless]
    reionization_model = 'ExpReionization'
    reion_exp_power = 2.0  # Exponent power for exponential reionization [dimensionless]
    lmax = 100  # Maximum multipole moment [dimensionless]

    # Set up CAMB parameters
    pars = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        omk=omk,
        tau=tau,
        As=As,
        ns=ns,
        reionization_model=reionization_model,
        reion_exp_power=reion_exp_power,
        WantTensors=False,
        lmax=lmax
    )

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in microKelvin^2
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax)
    cl_EE = powers['lensed_scalar'][:, 1]  # EE is column 1

    # Extract l=2 to l=100
    ls = np.arange(2, lmax + 1)
    EE = cl_EE[2:lmax + 1]

    # Prepare output directory and filename
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "result.csv")

    # Save to CSV
    output = np.column_stack((ls, EE))
    np.savetxt(output_path, output, delimiter=',', header='l,EE', fmt=['%d', '%.18e'], comments='')

    # Print summary to console
    print("CMB E-mode polarization power spectrum (Dl^EE = l(l+1)Cl^EE/2pi) computed for:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  tau = " + str(tau))
    print("  n_s = " + str(ns))
    print("  A_s = " + str(As) + " (dimensionless)")
    print("  Reionization model: " + reionization_model + " (exponent power = " + str(reion_exp_power) + ")")
    print("  Flat universe (Omega_k = 0)")
    print("Multipole range: l = 2 to " + str(lmax))
    print("Results saved to: " + output_path)
    print("First 5 rows (l, EE [microKelvin^2]):")
    for i in range(min(5, len(ls))):
        print("  " + str(ls[i]) + ", " + "{:.6e}".format(EE[i]))


if __name__ == "__main__":
    compute_cmb_ee_spectrum()