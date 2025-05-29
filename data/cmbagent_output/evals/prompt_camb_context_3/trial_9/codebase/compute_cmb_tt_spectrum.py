# filename: codebase/compute_cmb_tt_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_tt_spectrum():
    r"""
    Compute the raw (unlensed) CMB temperature power spectrum (C_l^{TT}) for a flat Lambda CDM cosmology
    using CAMB, with the following parameters:
        - Hubble constant (H0): 74 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Neutrino mass sum (mnu): 0.06 eV
        - Curvature (omk): 0
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965
    The spectrum is computed for multipole moments l=2 to l=3000, in units of microkelvin squared (\u00b5K^2).
    The results are saved to 'data/result.csv' with columns:
        - l: Multipole moment (integer, 2..3000)
        - TT: Raw temperature power spectrum (\u00b5K^2)
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set cosmological parameters
    pars = camb.set_params(
        H0=74.0,            # Hubble constant [km/s/Mpc]
        ombh2=0.022,        # Baryon density Ω_b h^2
        omch2=0.122,        # Cold dark matter density Ω_c h^2
        mnu=0.06,           # Neutrino mass sum [eV]
        omk=0.0,            # Curvature Ω_k
        tau=0.06,           # Optical depth to reionization
        As=2.0e-9,          # Scalar amplitude
        ns=0.965,           # Scalar spectral index
        lmax=3000,          # Maximum multipole
        WantTensors=False   # Only scalar modes
    )
    # Disable lensing to get the raw (unlensed) spectrum
    pars.DoLensing = False

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Extract the unlensed scalar power spectra in \u00b5K^2, raw Cl
    # Output: array of shape (lmax+1, 4), columns: TT, EE, BB, TE
    powers = results.get_unlensed_scalar_cls(CMB_unit='muK', raw_cl=True)

    # Prepare l and TT arrays for l=2..3000
    l_vals = np.arange(2, 3001)  # l=2..3000 inclusive
    TT_vals = powers[2:3001, 0]  # TT spectrum, \u00b5K^2

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'l': l_vals, 'TT': TT_vals})
    df.to_csv(output_path, index=False)

    # Print summary to console
    print("CMB raw temperature power spectrum (TT) computed and saved to " + output_path)
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())
    print("Total rows: " + str(df.shape[0]))
    print("l range: " + str(df['l'].min()) + " to " + str(df['l'].max()))
    print("TT units: microkelvin squared (\u00b5K^2)")


if __name__ == "__main__":
    compute_cmb_tt_spectrum()