# filename: codebase/compute_cmb_tt_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_tt_spectrum():
    r"""
    Compute the raw CMB temperature power spectrum (C_l^{TT}) for a flat Lambda CDM cosmology
    using CAMB, with the following parameters:
        - Hubble constant (H0): 70 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Neutrino mass sum (mnu): 0.06 eV
        - Curvature (omk): 0 (flat)
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965
    The function computes the lensed scalar TT power spectrum (C_l^{TT}) in units of μK^2
    for multipole moments l=2 to l=3000, and saves the results in a CSV file named 'result.csv'
    in the 'data/' directory, with columns:
        - l: Multipole moment (integer, 2 to 3000)
        - TT: Temperature power spectrum (C_l^{TT} in μK^2)
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set cosmological parameters and calculation settings
    pars = camb.set_params(
        H0=70.0,           # Hubble constant [km/s/Mpc]
        ombh2=0.022,       # Baryon density Ω_b h^2
        omch2=0.122,       # Cold dark matter density Ω_c h^2
        mnu=0.06,          # Neutrino mass sum [eV]
        omk=0.0,           # Curvature Ω_k (flat universe)
        tau=0.06,          # Optical depth to reionization
        As=2e-9,           # Scalar amplitude
        ns=0.965,          # Scalar spectral index
        lmax=3000,         # Maximum multipole for calculation
        WantScalars=True,
        WantTensors=False,
        WantVectors=False
    )

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Get the lensed scalar CMB power spectra (raw Cl, in μK^2)
    powers = results.get_cmb_power_spectra(
        params=pars,
        lmax=3000,
        spectra=['lensed_scalar'],
        CMB_unit='muK',
        raw_cl=True
    )

    # Extract TT spectrum for l=2 to l=3000
    Cl_TT_all = powers['lensed_scalar'][:, 0]  # TT is column 0
    l_values = np.arange(2, 3001)              # l = 2 to 3000
    Cl_TT_selected = Cl_TT_all[2:3001]         # Indices 2 to 3000 inclusive

    # Save to CSV
    df = pd.DataFrame({'l': l_values, 'TT': Cl_TT_selected})
    output_path = os.path.join(output_dir, "result.csv")
    df.to_csv(output_path, index=False)

    # Print summary
    print("CMB TT power spectrum (lensed, raw Cl, μK^2) computed for l=2 to l=3000.")
    print("Results saved to " + output_path)
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())

if __name__ == "__main__":
    compute_cmb_tt_spectrum()