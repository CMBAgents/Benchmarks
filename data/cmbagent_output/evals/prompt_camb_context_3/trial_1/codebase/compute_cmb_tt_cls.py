# filename: codebase/compute_cmb_tt_cls.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_tt_cls():
    r"""
    Compute the raw CMB temperature power spectrum (C_l^{TT}) for a flat Lambda CDM cosmology
    using CAMB, with the following parameters:
        - Hubble constant (H0): 74 km/s/Mpc
        - Baryon density (Omega_b h^2): 0.022
        - Cold dark matter density (Omega_c h^2): 0.122
        - Neutrino mass sum (Sigma m_nu): 0.06 eV
        - Curvature (Omega_k): 0
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (A_s): 2e-9
        - Scalar spectral index (n_s): 0.965

    The function computes the temperature power spectrum (C_l^{TT}) in units of microkelvin squared (μK^2)
    for multipole moments from l=2 to l=3000, and saves the results in a CSV file named 'result.csv'
    in the 'data/' directory, with columns:
        - l: Multipole moment (integer values from 2 to 3000)
        - TT: Raw temperature power spectrum (μK^2)
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
    # Optionally, ensure lensing is not computed (not needed for unlensed Cls)
    pars.DoLensing = False

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Get the raw unlensed scalar CMB temperature power spectrum in μK^2
    # Output: array of shape (lmax+1, 4), columns: TT, EE, BB, TE
    cls = results.get_unlensed_scalar_cls(lmax=lmax, CMB_unit='muK', raw_cl=True)

    # Extract l=2 to l=3000 and corresponding TT spectrum
    ls = np.arange(2, lmax + 1)  # l values from 2 to 3000
    cl_TT = cls[2:lmax + 1, 0]   # TT spectrum [μK^2]

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'l': ls, 'TT': cl_TT})
    df.to_csv(output_path, index=False)

    # Print summary to console
    print("CMB raw temperature power spectrum (C_l^{TT}) computed and saved to " + output_path)
    print("First 5 rows of the result:")
    print(df.head())
    print("Last 5 rows of the result:")
    print(df.tail())
    print("Total number of multipoles saved: " + str(len(df)))
    print("Units: TT is in microkelvin squared (μK^2), l is dimensionless.")


if __name__ == "__main__":
    compute_cmb_tt_cls()