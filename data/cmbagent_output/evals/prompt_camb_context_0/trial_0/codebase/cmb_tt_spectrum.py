# filename: codebase/cmb_tt_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_tt_spectrum():
    r"""
    Compute the CMB temperature power spectrum D_l^{TT} = l(l+1)C_l^{TT}/(2\pi) in units of muK^2
    for a flat Lambda CDM cosmology using CAMB, with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (Omega_b h^2): 0.02
        - Cold dark matter density (Omega_c h^2): 0.122
        - Neutrino mass sum (Sigma m_nu): 0.06 eV
        - Curvature (Omega_k): 0 (flat cosmology)
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (A_s): 2e-9
        - Scalar spectral index (n_s): 0.965

    The spectrum is computed for multipole moments l=2 to l=3000.
    The results are saved in 'data/result.csv' with columns:
        - l: Multipole moment (integer, 2 to 3000)
        - TT: Temperature power spectrum (muK^2)
    """
    # Cosmological parameters
    H0 = 67.5  # [km/s/Mpc]
    ombh2 = 0.02  # [dimensionless]
    omch2 = 0.122  # [dimensionless]
    mnu = 0.06  # [eV]
    omk = 0.0  # [dimensionless]
    tau = 0.06  # [dimensionless]
    As = 2.0e-9  # [dimensionless]
    ns = 0.965  # [dimensionless]
    lmax_calc = 3000  # [dimensionless]

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
        lmax=lmax_calc,
        WantScalars=True,
        WantTensors=False,
        DoLensing=True
    )

    # Run CAMB and get results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in muK^2
    # Output: array of shape (lmax+1, 4), columns: TT, EE, BB, TE
    powers = results.get_lensed_scalar_cls(lmax=lmax_calc, CMB_unit='muK')

    # Extract l and TT spectrum for l=2 to l=3000
    ls = np.arange(2, lmax_calc + 1)  # [dimensionless]
    Dl_TT = powers[2:lmax_calc + 1, 0]  # [muK^2]

    # Prepare output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to CSV
    csv_filename = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'l': ls, 'TT': Dl_TT})
    df.to_csv(csv_filename, index=False)

    # Print summary to console
    print("CMB temperature power spectrum (lensed, TT) saved to " + csv_filename)
    print("Columns: l (multipole, dimensionless), TT (muK^2)")
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())
    print("Total number of multipoles saved: " + str(len(df)))


if __name__ == "__main__":
    compute_cmb_tt_spectrum()