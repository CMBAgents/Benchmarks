# filename: codebase/compute_cmb_ee_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_ee_spectrum(
    H0=67.5,
    ombh2=0.022,
    omch2=0.122,
    tau=0.1,
    ns=0.95,
    As_base=1.8e-9,
    reion_model='ExpReionization',
    reion_exp_power=2,
    lmin=2,
    lmax=100,
    lmax_calc=250,
    output_folder='data',
    output_filename='result.csv'
):
    """
    Compute the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for a flat Lambda CDM cosmology
    using CAMB, with exponential reionization (power=2), and save l=2..100 to CSV.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Baryon density parameter (Omega_b h^2).
    omch2 : float
        Cold dark matter density parameter (Omega_c h^2).
    tau : float
        Optical depth to reionization (dimensionless).
    ns : float
        Scalar spectral index (dimensionless).
    As_base : float
        Base scalar amplitude (dimensionless).
    reion_model : str
        Reionization model name ('ExpReionization').
    reion_exp_power : float
        Exponent power for exponential reionization.
    lmin : int
        Minimum multipole moment to output (inclusive).
    lmax : int
        Maximum multipole moment to output (inclusive).
    lmax_calc : int
        Maximum multipole for CAMB calculation (should be >= lmax for accuracy).
    output_folder : str
        Folder to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
    """
    # Calculate As = As_base * exp(2 * tau)
    As = As_base * np.exp(2.0 * tau)

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Set up CAMB parameters
    pars = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        omk=0.0,
        tau=tau,
        ns=ns,
        As=As,
        reionization_model=reion_model,
        reion_exp_power=reion_exp_power,
        lmax=lmax_calc,
        WantScalars=True,
        WantTensors=False
    )

    # Run CAMB
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in muK^2
    powers = results.get_cmb_power_spectra(lmax=lmax, CMB_unit='muK', spectra=['lensed_scalar'])
    cl_lensed_scalar = powers['lensed_scalar']  # shape (lmax+1, 4)

    # Extract EE spectrum for l=lmin..lmax
    ls = np.arange(cl_lensed_scalar.shape[0])  # 0..lmax
    EE = cl_lensed_scalar[lmin:lmax+1, 1]      # EE column, l=lmin..lmax
    lvals = ls[lmin:lmax+1]

    # Save to CSV
    output_path = os.path.join(output_folder, output_filename)
    output = pd.DataFrame({'l': lvals, 'EE': EE})
    output.to_csv(output_path, index=False)

    # Print results
    print("CMB E-mode power spectrum (lensed, l(l+1)C_l^{EE}/(2pi)) saved to " + output_path)
    print("Multipole range: l = " + str(lmin) + " to l = " + str(lmax))
    print("EE units: microkelvin^2 (muK^2)")
    print("Cosmological parameters used:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  tau = " + str(tau))
    print("  ns = " + str(ns))
    print("  As = " + str(As) + " (calculated as 1.8e-9 * exp(2*tau))")
    print("  Reionization model: " + reion_model + " (exponent power = " + str(reion_exp_power) + ")")
    print("First 5 rows of output:")
    print(output.head())


if __name__ == "__main__":
    compute_cmb_ee_spectrum()