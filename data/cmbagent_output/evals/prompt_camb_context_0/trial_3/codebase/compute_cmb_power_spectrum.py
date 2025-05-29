# filename: codebase/compute_cmb_power_spectrum.py
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
    lmax=3000,         # Maximum multipole [dimensionless]
    output_dir="data/",# Output directory for results
    output_csv="result.csv" # Output CSV filename
):
    r"""
    Compute the CMB temperature power spectrum l(l+1)C_l^{TT}/(2pi) in units of muK^2
    for a flat Lambda CDM cosmology using CAMB, and save the results to a CSV file.

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
    output_dir : str
        Directory to save the output CSV file.
    output_csv : str
        Name of the output CSV file.

    Returns
    -------
    None
        The function saves the results to a CSV file and prints a summary to the console.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
        lmax=lmax,
        halofit_version='mead'
    )

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Get CMB power spectra in muK^2 units, D_ell = l(l+1)C_ell/2pi
    powers = results.get_cmb_power_spectra(CMB_unit='muK')
    total_cls = powers['total']  # shape (lmax+1, 4): columns are TT, EE, BB, TE

    # Multipole moments l = 0, 1, ..., lmax
    ls = np.arange(total_cls.shape[0])  # [0, 1, ..., lmax]

    # Extract l=2 to l=lmax (inclusive)
    l_start = 2
    l_end = lmax
    l_values = ls[l_start:l_end+1]
    tt_spectrum = total_cls[l_start:l_end+1, 0]  # TT spectrum [muK^2]

    # Save to CSV
    output_path = os.path.join(output_dir, output_csv)
    df = pd.DataFrame({'l': l_values.astype(int), 'TT': tt_spectrum})
    df.to_csv(output_path, index=False)

    # Print summary to console
    print("CMB temperature power spectrum (l(l+1)C_l^{TT}/2pi in muK^2) computed for l=2 to l=" + str(lmax))
    print("Results saved to " + output_path)
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())


if __name__ == "__main__":
    compute_cmb_tt_spectrum()