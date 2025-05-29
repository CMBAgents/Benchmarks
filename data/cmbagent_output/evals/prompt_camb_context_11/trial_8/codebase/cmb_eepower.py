# filename: codebase/cmb_eepower.py
import os
import numpy as np
import pandas as pd
import camb


def compute_cmb_ee_spectrum(
    H0=67.5,
    ombh2=0.022,
    omch2=0.122,
    tau=0.1,
    ns=0.95,
    As_base=1.8e-9,
    reion_exp_power=2,
    lmin=2,
    lmax=100,
    lmax_calc=250,
    output_dir="data",
    output_filename="result.csv"
):
    """
    Compute the CMB E-mode polarization power spectrum l(l+1)C_l^{EE}/(2\pi) in μK^2
    for a flat Lambda CDM cosmology using CAMB, with exponential reionization (power=2),
    and save the results for l = lmin to l = lmax in output_dir/output_filename.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter (Omega_b h^2).
    omch2 : float
        Physical cold dark matter density parameter (Omega_c h^2).
    tau : float
        Optical depth to reionization.
    ns : float
        Scalar spectral index.
    As_base : float
        Base scalar amplitude (before tau correction).
    reion_exp_power : float
        Exponent power for exponential reionization model.
    lmin : int
        Minimum multipole to output (inclusive).
    lmax : int
        Maximum multipole to output (inclusive).
    lmax_calc : int
        Maximum multipole for CAMB calculation (should be >= lmax for accuracy).
    output_dir : str
        Directory to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
    """
    # Calculate As with tau correction
    As = As_base * np.exp(2 * tau)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up CAMB parameters
    pars = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        omk=0.0,
        tau=tau,
        As=As,
        ns=ns,
        reionization_model='ExpReionization',
        reion_exp_power=reion_exp_power,
        lmax=lmax_calc,
        WantScalars=True,
        WantTensors=False
    )

    # Run CAMB
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in μK^2
    powers = results.get_cmb_power_spectra(
        lmax=lmax,
        CMB_unit='muK',
        spectra=['lensed_scalar'],
        raw_cl=False
    )
    cl_lensed_scalar = powers['lensed_scalar']  # shape (lmax+1, 4)
    EE = cl_lensed_scalar[:, 1]  # EE is column 1
    ls = np.arange(cl_lensed_scalar.shape[0])

    # Extract l = lmin to l = lmax
    l_vals = ls[lmin:lmax+1]
    EE_vals = EE[lmin:lmax+1]

    # Save to CSV
    output_path = os.path.join(output_dir, output_filename)
    output = pd.DataFrame({'l': l_vals, 'EE': EE_vals})
    output.to_csv(output_path, index=False)

    # Print results
    print("CMB E-mode power spectrum l(l+1)C_l^{EE}/(2pi) in muK^2 saved to " + output_path)
    print("Cosmological parameters used:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  ombh2 = " + str(ombh2))
    print("  omch2 = " + str(omch2))
    print("  tau = " + str(tau))
    print("  ns = " + str(ns))
    print("  As = " + str(As) + " (calculated as 1.8e-9 * exp(2 * tau))")
    print("  Reionization model: Exponential, power = " + str(reion_exp_power))
    print("  Output multipole range: l = " + str(lmin) + " to l = " + str(lmax))
    print("First 5 rows of the output:")
    print(output.head())


if __name__ == "__main__":
    compute_cmb_ee_spectrum()