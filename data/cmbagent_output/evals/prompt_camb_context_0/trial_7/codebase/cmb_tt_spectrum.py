# filename: codebase/cmb_tt_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_tt_spectrum():
    r"""
    Compute the CMB temperature power spectrum for a flat Lambda CDM cosmology using CAMB.

    Cosmological parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (Omega_b h^2): 0.02
        - Cold dark matter density (Omega_c h^2): 0.122
        - Neutrino mass sum (sum m_nu): 0.06 eV
        - Curvature (Omega_k): 0 (flat)
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (A_s): 2e-9
        - Scalar spectral index (n_s): 0.965

    Computes the lensed temperature power spectrum l(l+1)C_l^{TT}/(2pi) in units of muK^2
    for multipole moments l=2 to l=3000, and saves the results in 'data/result.csv'.

    Returns
    -------
    None
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set cosmological parameters
    pars = camb.set_params(
        H0=67.5,           # Hubble constant [km/s/Mpc]
        ombh2=0.02,        # Omega_b h^2 (baryon density)
        omch2=0.122,       # Omega_c h^2 (cold dark matter density)
        mnu=0.06,          # Sum of neutrino masses [eV]
        omk=0,             # Curvature (flat universe)
        tau=0.06,          # Optical depth to reionization
        As=2e-9,           # Scalar amplitude
        ns=0.965,          # Scalar spectral index
        lmax=3000,         # Maximum multipole
        WantTensors=False  # Only scalar modes
    )

    # Run CAMB and get results
    results = camb.get_results(pars)

    # Get lensed CMB power spectra in muK^2 units
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=False)
    total_cls = powers['total']  # columns: TT, EE, BB, TE

    # Multipole moments (l) corresponding to the rows of total_cls
    ls = np.arange(total_cls.shape[0])

    # Select l=2 to l=3000 (inclusive)
    lmin, lmax = 2, 3000
    if lmax > ls[-1]:
        print("Requested lmax (" + str(lmax) + ") exceeds computed lmax (" + str(ls[-1]) + "). Truncating to computed lmax.")
        lmax = ls[-1]
    lvals = ls[lmin:lmax+1]
    ttvals = total_cls[lmin:lmax+1, 0]  # TT spectrum in muK^2

    # Save to CSV
    output_file = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'l': lvals.astype(int), 'TT': ttvals})
    df.to_csv(output_file, index=False)

    # Print summary to console
    print("CMB temperature power spectrum (lensed, TT) saved to " + output_file)
    print("Columns:")
    print("l: Multipole moment (dimensionless)")
    print("TT: l(l+1)C_l^{TT}/(2pi) [muK^2]")
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())
    print("Total number of rows: " + str(df.shape[0]))

if __name__ == "__main__":
    compute_cmb_tt_spectrum()
