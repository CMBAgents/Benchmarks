# filename: codebase/compute_cmb_tt_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_tt_spectrum(
    H0=67.5,           # Hubble constant [km/s/Mpc]
    ombh2=0.02,        # Baryon density Omega_b h^2 [dimensionless]
    omch2=0.122,       # Cold dark matter density Omega_c h^2 [dimensionless]
    mnu=0.06,          # Neutrino mass sum [eV]
    omk=0.0,           # Curvature Omega_k [dimensionless]
    tau=0.06,          # Optical depth to reionization [dimensionless]
    As=2e-9,           # Scalar amplitude [dimensionless]
    ns=0.965,          # Scalar spectral index [dimensionless]
    lmax=3000,         # Maximum multipole moment [dimensionless]
    lmin_output=2,     # Minimum multipole moment for output [dimensionless]
    lmax_output=3000,  # Maximum multipole moment for output [dimensionless]
    output_dir="data/",
    output_filename="result.csv"
):
    r"""
    Compute the CMB temperature power spectrum for a flat Lambda CDM cosmology using CAMB.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Baryon density Omega_b h^2 (dimensionless).
    omch2 : float
        Cold dark matter density Omega_c h^2 (dimensionless).
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature Omega_k (dimensionless).
    tau : float
        Optical depth to reionization (dimensionless).
    As : float
        Scalar amplitude (dimensionless).
    ns : float
        Scalar spectral index (dimensionless).
    lmax : int
        Maximum multipole moment to compute (dimensionless).
    lmin_output : int
        Minimum multipole moment to output (dimensionless).
    lmax_output : int
        Maximum multipole moment to output (dimensionless).
    output_dir : str
        Directory to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
        Saves the results to a CSV file and prints a summary to the console.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set cosmological parameters
    params = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu,
        omk=omk,
        tau=tau,
        As=As,
        ns=ns,
        lmax=lmax,
        WantTensors=False
    )

    # Run CAMB to get results
    results = camb.get_results(params)

    # Get lensed CMB power spectra in muK^2 units
    powers = results.get_cmb_power_spectra(params, CMB_unit='muK', raw_cl=False)
    total_cls = powers['total']  # columns: TT, EE, BB, TE

    # Multipole moments (l) corresponding to the rows of total_cls
    ls = np.arange(total_cls.shape[0])

    # Ensure lmax_output does not exceed computed lmax
    if lmax_output > ls[-1]:
        print("Requested lmax_output (" + str(lmax_output) + ") exceeds computed lmax (" + str(ls[-1]) + "). Truncating to computed lmax.")
        lmax_output = ls[-1]

    # Extract l and TT spectrum for lmin_output <= l <= lmax_output
    l_vals = ls[lmin_output:lmax_output+1]
    tt_vals = total_cls[lmin_output:lmax_output+1, 0]  # TT spectrum [muK^2]

    # Save to CSV
    output_path = os.path.join(output_dir, output_filename)
    df = pd.DataFrame({'l': l_vals.astype(int), 'TT': tt_vals})
    df.to_csv(output_path, index=False)

    # Print summary to console
    print("CMB temperature power spectrum (lensed, TT) computed for flat Lambda CDM cosmology.")
    print("Cosmological parameters used:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  Sum m_nu = " + str(mnu) + " eV")
    print("  Omega_k = " + str(omk))
    print("  tau = " + str(tau))
    print("  A_s = " + str(As))
    print("  n_s = " + str(ns))
    print("Multipole range: l = " + str(lmin_output) + " to " + str(lmax_output))
    print("Results saved to: " + output_path)
    print("First 5 rows of the output:")
    print(df.head().to_string(index=False))
    print("Last 5 rows of the output:")
    print(df.tail().to_string(index=False))


if __name__ == "__main__":
    compute_cmb_tt_spectrum()