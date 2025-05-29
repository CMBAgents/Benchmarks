# filename: codebase/compute_cmb_tt_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_tt_spectrum(
    H0=67.5,           # Hubble constant [km/s/Mpc]
    ombh2=0.02,        # Baryon density [dimensionless]
    omch2=0.122,       # Cold dark matter density [dimensionless]
    mnu=0.06,          # Sum of neutrino masses [eV]
    omk=0.0,           # Curvature [dimensionless]
    tau=0.06,          # Optical depth to reionization [dimensionless]
    As=2e-9,           # Scalar amplitude [dimensionless]
    ns=0.965,          # Scalar spectral index [dimensionless]
    lmax=3000,         # Maximum multipole moment [dimensionless]
    output_dir="data/",
    output_filename="result.csv"
):
    r"""
    Compute the CMB temperature power spectrum (lensed scalar TT) for a flat Lambda CDM cosmology
    using CAMB, and save the results to a CSV file.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter.
    omch2 : float
        Physical cold dark matter density parameter.
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter (0 for flat).
    tau : float
        Optical depth to reionization.
    As : float
        Scalar amplitude of primordial power spectrum.
    ns : float
        Scalar spectral index.
    lmax : int
        Maximum multipole moment to compute.
    output_dir : str
        Directory to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
        The function saves the results to a CSV file and prints a summary to the console.

    Output File Format
    ------------------
    CSV file with columns:
        - l: Multipole moment (integer, 2 to lmax)
        - TT: Temperature power spectrum (l(l+1)C_l^{TT}/(2pi)) in microkelvin squared (muK^2)
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
        lmax=lmax
    )

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in muK^2 units
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax)
    lensed_scalar_cls = powers['lensed_scalar']  # shape: (lmax+1, 4), columns: TT, EE, BB, TE

    # Extract TT spectrum for l=2 to lmax
    ls = np.arange(2, lmax + 1)  # l values
    TT_values = lensed_scalar_cls[ls, 0]  # TT spectrum in muK^2

    # Save to CSV
    output_path = os.path.join(output_dir, output_filename)
    df = pd.DataFrame({'l': ls, 'TT': TT_values})
    df.to_csv(output_path, index=False)

    # Print summary to console
    print("CMB temperature power spectrum (lensed scalar TT) computed for flat Lambda CDM cosmology.")
    print("Cosmological parameters used:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  Sum m_nu = " + str(mnu) + " eV")
    print("  Omega_k = " + str(omk))
    print("  tau = " + str(tau))
    print("  A_s = " + str(As))
    print("  n_s = " + str(ns))
    print("  lmax = " + str(lmax))
    print("\nResults saved to: " + output_path)
    print("Columns: l (multipole moment, 2-" + str(lmax) + "), TT (l(l+1)C_l^{TT}/(2pi) in muK^2)")
    print("\nSample of results:")
    print(df.head(10).to_string(index=False))
    print("...")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    compute_cmb_tt_spectrum()