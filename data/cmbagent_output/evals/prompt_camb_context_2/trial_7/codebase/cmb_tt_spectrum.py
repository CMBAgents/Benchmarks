# filename: codebase/cmb_tt_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_tt_spectrum():
    """
    Compute the lensed scalar CMB temperature power spectrum (C_l^{TT}) for a flat Lambda CDM cosmology
    using CAMB, for l=2 to l=3000, and save the results to data/result.csv.

    Returns
    -------
    None
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set cosmological parameters (units in comments)
    pars = camb.set_params(
        H0=70.0,           # Hubble constant [km/s/Mpc]
        ombh2=0.022,       # Baryon density Ω_b h^2 [dimensionless]
        omch2=0.122,       # Cold dark matter density Ω_c h^2 [dimensionless]
        mnu=0.06,          # Neutrino mass sum [eV]
        omk=0,             # Curvature Ω_k [dimensionless]
        tau=0.06,          # Optical depth to reionization [dimensionless]
        As=2e-9,           # Scalar amplitude [dimensionless]
        ns=0.965,          # Scalar spectral index [dimensionless]
        lmax=3000,         # Maximum multipole moment
        WantTensors=False,
        WantScalars=True,
        DoLensing=True
    )

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Get the lensed scalar CMB power spectra in μK^2, raw Cl
    powers = results.get_cmb_power_spectra(
        params=pars,
        CMB_unit="muK",
        raw_cl=True,
        spectra=["lensed_scalar"]
    )

    # Extract TT spectrum (column 0) for l=2 to l=3000
    cl_tt = powers["lensed_scalar"][:, 0]  # shape: (lmax+1,)
    l_arr = np.arange(2, 3001)             # l = 2..3000
    tt_arr = cl_tt[2:3001]                 # C_l^{TT} in μK^2

    # Save to CSV
    df = pd.DataFrame({"l": l_arr, "TT": tt_arr})
    output_path = os.path.join(output_dir, "result.csv")
    df.to_csv(output_path, index=False)

    # Print summary to console
    print("Raw CMB temperature power spectrum (lensed scalar C_l^{TT}) saved to " + output_path)
    print("First five rows:")
    print(df.head())
    print("Last five rows:")
    print(df.tail())
    print("Total number of multipoles saved: " + str(len(df)))
    print("TT units: microkelvin^2 (μK^2)")


if __name__ == "__main__":
    compute_cmb_tt_spectrum()
