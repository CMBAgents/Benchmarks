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
        - Curvature (omk): 0
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965
    The spectrum is computed for multipole moments l=2 to l=3000, in units of microkelvin squared (muK^2).
    The results are saved to 'data/result.csv' with columns:
        - l: Multipole moment (integer, 2..3000)
        - TT: Temperature power spectrum (C_l^{TT} in muK^2)
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set cosmological parameters
    pars = camb.set_params(
        H0=70.0,           # Hubble constant [km/s/Mpc]
        ombh2=0.022,       # Baryon density [dimensionless]
        omch2=0.122,       # Cold dark matter density [dimensionless]
        mnu=0.06,          # Neutrino mass sum [eV]
        omk=0.0,           # Curvature [dimensionless]
        tau=0.06,          # Optical depth to reionization [dimensionless]
        As=2e-9,           # Scalar amplitude [dimensionless]
        ns=0.965,          # Scalar spectral index [dimensionless]
        lmax=3000,         # Maximum multipole moment
        WantTensors=False, # Only scalar spectra
        WantScalars=True,
        DoLensing=True     # Compute lensed spectra
    )

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Get the lensed scalar CMB power spectra in muK^2, raw Cl
    powers = results.get_cmb_power_spectra(
        params=pars,
        CMB_unit='muK',
        raw_cl=True,
        spectra=['lensed_scalar']
    )
    # Extract TT spectrum (column 0)
    cl_tt = powers['lensed_scalar'][:, 0]  # [muK^2]

    # Prepare l and TT arrays for l=2..3000
    ls = np.arange(pars.max_l + 1)  # l=0..3000
    l_out = ls[2:3001]               # l=2..3000
    tt_out = cl_tt[2:3001]           # C_l^{TT} for l=2..3000

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'l': l_out, 'TT': tt_out})
    df.to_csv(output_path, index=False)

    # Print summary to console
    print("Raw CMB temperature power spectrum (lensed scalar C_l^TT) saved to " + output_path)
    print("First five rows:")
    print(df.head())
    print("Last five rows:")
    print(df.tail())
    print("Total number of multipoles saved: " + str(len(df)))


if __name__ == "__main__":
    compute_cmb_tt_spectrum()