# filename: codebase/delensed_cmb_tt.py
import camb
import numpy as np
import pandas as pd
import os


def compute_delensed_cmb_tt_spectrum():
    r"""
    Compute the delensed CMB temperature power spectrum l(l+1)C_l^{TT}/(2pi) for a flat Lambda CDM cosmology
    using CAMB, with specified cosmological parameters and 80% delensing efficiency.
    The result is saved as a CSV file with columns:
        l: Multipole moment (integer, 2 to 3000)
        TT: Delensed temperature power spectrum (l(l+1)C_l^{TT}/(2pi)) in micro-Kelvin^2
    The file is saved as 'data/result.csv'.

    Units:
        - H0: km/s/Mpc
        - ombh2: dimensionless (Omega_b h^2)
        - omch2: dimensionless (Omega_c h^2)
        - mnu: eV (sum of neutrino masses)
        - omk: dimensionless (curvature)
        - tau: dimensionless (optical depth)
        - As: dimensionless (scalar amplitude)
        - ns: dimensionless (scalar spectral index)
        - TT: micro-Kelvin^2
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density Omega_b h^2
    omch2 = 0.122  # Cold dark matter density Omega_c h^2
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature Omega_k
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index

    # Delensing efficiency and Alens
    delensing_efficiency = 0.8  # 80% delensing
    Alens_residual = 1.0 - delensing_efficiency  # Fraction of lensing potential remaining

    # Multipole range
    lmin = 2
    lmax = 3000
    lmax_calc = 3500  # Set calculation lmax higher for accuracy

    # Set CAMB parameters
    pars = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu,
        omk=omk,
        tau=tau,
        As=As,
        ns=ns
    )
    pars.set_for_lmax(lmax=lmax_calc, lens_potential_accuracy=1)

    # Run CAMB
    results = camb.get_results(pars)

    # Get delensed CMB power spectra (TT, EE, BB, TE)
    # Output: array of shape (lmax+1, 4), columns: TT, EE, BB, TE
    cls = results.get_partially_lensed_cls(
        Alens=Alens_residual,
        lmax=lmax,
        CMB_unit='muK',
        raw_cl=False  # Returns l(l+1)C_l/2pi
    )

    # Extract multipole moments and TT spectrum
    ls = np.arange(lmin, lmax + 1)  # l = 2..3000
    TT = cls[lmin:lmax + 1, 0]  # TT is the first column, units: micro-Kelvin^2

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'l': ls, 'TT': TT})
    df.to_csv(output_path, index=False)

    # Print summary
    print("Delensed CMB TT power spectrum computed and saved to " + output_path)
    print("Columns:")
    print("  l  : Multipole moment (integer, 2 to 3000)")
    print("  TT : Delensed temperature power spectrum (l(l+1)C_l^{TT}/(2pi)) in micro-Kelvin^2")
    print("First 5 rows of the result:")
    print(df.head())
    print("Total number of rows: " + str(len(df)))


if __name__ == "__main__":
    compute_delensed_cmb_tt_spectrum()