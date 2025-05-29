# filename: codebase/cmb_tt_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_tt_spectrum():
    r"""
    Compute the lensed CMB temperature power spectrum D_l^{TT} = l(l+1)C_l^{TT}/(2pi) in muK^2
    for a non-flat Lambda CDM cosmology using CAMB, with the following parameters:
        - Hubble constant (H0): 67.3 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Neutrino mass sum (mnu): 0.06 eV
        - Curvature (omk): 0.05
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965
    The spectrum is computed for multipole moments l=2 to l=3000.
    The results are saved in 'data/result.csv' with columns:
        - l: Multipole moment (integer, 2 to 3000)
        - TT: Temperature power spectrum (muK^2)
    """
    # Cosmological parameters (units in comments)
    H0 = 67.3                # Hubble constant [km/s/Mpc]
    ombh2 = 0.022            # Omega_b h^2 [dimensionless]
    omch2 = 0.122            # Omega_c h^2 [dimensionless]
    mnu = 0.06               # Sum of neutrino masses [eV]
    omk = 0.05               # Curvature parameter [dimensionless]
    tau = 0.06               # Optical depth [dimensionless]
    As = 2.0e-9              # Scalar amplitude [dimensionless]
    ns = 0.965               # Scalar spectral index [dimensionless]
    l_max_calc = 3000        # Maximum multipole [dimensionless]

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
        lmax=l_max_calc,
        lens_potential_accuracy=1,
        WantScalars=True,
        WantTensors=False
    )

    # Run CAMB and get results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in muK^2 units
    powers = results.get_cmb_power_spectra(params=pars, CMB_unit='muK', spectra=['lensed_scalar'])
    lensed_cls = powers['lensed_scalar']  # shape: (lmax+1, 4), columns: TT, EE, BB, TE

    # Extract TT spectrum for l=2 to l=3000
    l_values = np.arange(2, l_max_calc + 1)  # l = 2..3000
    tt_spectrum_Dl = lensed_cls[2:l_max_calc + 1, 0]  # TT column, D_l^{TT} in muK^2

    # Prepare output DataFrame
    df = pd.DataFrame({'l': l_values, 'TT': tt_spectrum_Dl})

    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    df.to_csv(output_path, index=False)

    # Print summary to console
    print("CMB lensed TT power spectrum (D_l = l(l+1)C_l/(2pi)) in muK^2 saved to " + output_path)
    print("First 5 rows of the result:")
    print(df.head())
    print("Last 5 rows of the result:")
    print(df.tail())
    print("Total number of multipoles saved: " + str(len(df)))


if __name__ == "__main__":
    compute_cmb_tt_spectrum()