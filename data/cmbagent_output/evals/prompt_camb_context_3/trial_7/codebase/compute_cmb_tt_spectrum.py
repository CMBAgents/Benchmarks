# filename: codebase/compute_cmb_tt_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_tt_spectrum():
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

    The spectrum is computed for multipole moments l=2 to l=3000, in units of microkelvin squared (muK^2).
    The results are saved to 'data/result.csv' with columns:
        - l: Multipole moment (integer, 2 to 3000)
        - TT: Raw temperature power spectrum (muK^2)
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cosmological parameters
    H0 = 74.0  # [km/s/Mpc]
    ombh2 = 0.022  # [dimensionless]
    omch2 = 0.122  # [dimensionless]
    mnu = 0.06  # [eV]
    omk = 0.0  # [dimensionless]
    tau = 0.06  # [dimensionless]
    As = 2.0e-9  # [dimensionless]
    ns = 0.965  # [dimensionless]
    lmax = 3000  # [dimensionless]

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

    # Get the raw unlensed scalar CMB TT power spectrum in muK^2
    # Output: array of shape (lmax+1, 4), columns: TT, EE, BB, TE
    cls = results.get_unlensed_scalar_cls(lmax=lmax, CMB_unit='muK', raw_cl=True)

    # Extract l=2 to l=3000
    ls = np.arange(2, lmax + 1)  # [dimensionless]
    TT = cls[2:lmax + 1, 0]  # [muK^2]

    # Save to CSV
    output_file = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'l': ls, 'TT': TT})
    df.to_csv(output_file, index=False)

    # Print summary to console
    print("CMB raw temperature power spectrum (C_l^{TT}) computed for l=2 to l=3000.")
    print("Results saved to " + output_file)
    print("First five rows of the result:")
    print(df.head())

if __name__ == "__main__":
    compute_cmb_tt_spectrum()