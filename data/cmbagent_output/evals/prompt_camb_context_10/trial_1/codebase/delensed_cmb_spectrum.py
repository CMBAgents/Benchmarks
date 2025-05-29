# filename: codebase/delensed_cmb_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_and_save_delensed_cmb_tt_spectrum():
    r"""
    Compute the delensed CMB temperature power spectrum l(l+1)C_l^{TT}/(2\pi) in units of muK^2
    for a flat Lambda CDM cosmology with specified parameters, applying 80% delensing efficiency.
    Save the results for l=2 to l=3000 in a CSV file with columns:
        - l: Multipole moment (integer, 2 to 3000)
        - TT: Delensed temperature power spectrum (muK^2)
    The file is saved as 'data/result.csv'.

    Cosmological parameters:
        H0: 67.5 km/s/Mpc
        ombh2: 0.022
        omch2: 0.122
        mnu: 0.06 eV
        omk: 0.0
        tau: 0.06
        As: 2e-9
        ns: 0.965
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    As = 2e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]
    lmax = 3000  # Maximum multipole moment [dimensionless]
    lens_potential_accuracy = 1  # Planck-level accuracy

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
        lens_potential_accuracy=lens_potential_accuracy
    )

    # Compute results
    results = camb.get_results(pars)

    # Delensing efficiency: 80% (residual lensing Alens=0.2)
    delensing_efficiency = 0.8  # [dimensionless]
    Alens_residual = 1.0 - delensing_efficiency  # [dimensionless]

    # Get delensed CMB power spectra in muK^2, l(l+1)C_l/2pi
    delensed_cls = results.get_partially_lensed_cls(
        Alens=Alens_residual,
        lmax=lmax,
        CMB_unit='muK',
        raw_cl=False
    )

    # Extract l and TT spectrum for l=2 to l=3000
    ls = np.arange(2, lmax + 1)  # [dimensionless]
    tt = delensed_cls[2:lmax + 1, 0]  # [muK^2]

    # Save to CSV
    output = pd.DataFrame({'l': ls.astype(int), 'TT': tt})
    csv_path = os.path.join(output_dir, "result.csv")
    output.to_csv(csv_path, index=False)

    # Print summary to console
    pd.set_option("display.precision", 8)
    pd.set_option("display.width", 120)
    print("Delensed CMB TT power spectrum (l(l+1)C_l^{TT}/(2pi)) in muK^2 saved to " + csv_path)
    print("First 5 rows:")
    print(output.head())
    print("Last 5 rows:")
    print(output.tail())
    print("Total number of multipoles saved: " + str(len(output)))


if __name__ == "__main__":
    compute_and_save_delensed_cmb_tt_spectrum()