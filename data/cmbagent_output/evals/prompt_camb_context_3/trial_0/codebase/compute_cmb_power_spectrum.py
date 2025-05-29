# filename: codebase/compute_cmb_power_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_tt_cls():
    r"""
    Compute the raw CMB temperature power spectrum (C_l^{TT}) for a flat Lambda CDM cosmology
    using CAMB, with the following parameters:
        - Hubble constant (H0): 74 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Neutrino mass sum (mnu): 0.06 eV
        - Curvature (omk): 0
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965

    The function computes the temperature power spectrum (C_l^{TT}) in units of μK^2 for
    multipole moments from l=2 to l=3000, and saves the results in a CSV file named 'result.csv'
    with two columns:
        - l: Multipole moment (integer values from 2 to 3000)
        - TT: Raw temperature power spectrum (μK^2)

    The output file is saved in the 'data/' directory.
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cosmological parameters
    H0 = 74.0  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density Omega_b h^2 [dimensionless]
    omch2 = 0.122  # Cold dark matter density Omega_c h^2 [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature Omega_k [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    As = 2.0e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]
    lmax = 3000  # Maximum multipole moment

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
        WantTensors=False
    )

    # Run CAMB and get results
    results = camb.get_results(pars)

    # Get raw unlensed scalar CMB power spectra in μK^2
    # Output: array of shape (lmax+1, 4), columns: TT, EE, BB, TE
    cls = results.get_unlensed_scalar_cls(lmax=lmax, CMB_unit='muK', raw_cl=True)

    # Extract l=2 to l=3000 and corresponding TT spectrum
    ls = np.arange(2, lmax + 1)  # l values [2, 3, ..., 3000]
    TT = cls[2:, 0]  # TT spectrum [μK^2] for l=2 to l=3000

    # Save to CSV
    output_file = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'l': ls, 'TT': TT})
    df.to_csv(output_file, index=False)

    # Print summary to console
    print("CMB raw temperature power spectrum (C_l^{TT}) saved to " + output_file)
    print("First five rows of the result:")
    print(df.head())
    print("\nUnits: l (dimensionless), TT (μK^2)")


if __name__ == "__main__":
    compute_cmb_tt_cls()