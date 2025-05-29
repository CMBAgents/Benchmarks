# filename: codebase/compute_cmb_bmode_power_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_bmode_power_spectrum():
    r"""
    Compute the CMB B-mode polarization power spectrum D_l^{BB} = l(l+1)C_l^{BB}/(2\pi)
    in units of microkelvin squared (muK^2) for a flat Lambda CDM cosmology using CAMB.

    Cosmological parameters:
        Hubble constant (H0): 67.5 km/s/Mpc
        Baryon density (ombh2): 0.022
        Cold dark matter density (omch2): 0.122
        Neutrino mass sum (mnu): 0.06 eV
        Curvature (omk): 0
        Optical depth to reionization (tau): 0.06
        Tensor-to-scalar ratio (r): 0
        Scalar amplitude (As): 2e-9
        Scalar spectral index (ns): 0.965

    The function computes the lensed scalar CMB power spectra up to l=3000,
    extracts the B-mode (BB) spectrum for l=2 to l=3000, and saves the results
    in a CSV file named 'result.csv' in the 'data/' directory, with columns:
        l: Multipole moment (integer, 2 to 3000)
        BB: B-mode power spectrum (D_l^{BB} in muK^2)

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
        H0=67.5,                # Hubble constant [km/s/Mpc]
        ombh2=0.022,            # Baryon density [dimensionless]
        omch2=0.122,            # Cold dark matter density [dimensionless]
        mnu=0.06,               # Neutrino mass sum [eV]
        omk=0.0,                # Curvature [dimensionless]
        tau=0.06,               # Optical depth to reionization [dimensionless]
        As=2e-9,                # Scalar amplitude [dimensionless]
        ns=0.965,               # Scalar spectral index [dimensionless]
        r=0.0,                  # Tensor-to-scalar ratio [dimensionless]
        lmax=3000,              # Maximum multipole moment
        lens_potential_accuracy=1 # Lensing accuracy (important for B-modes)
    )

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in muK^2 units
    # Returns array of shape (lmax+1, 4): columns are TT, EE, BB, TE
    powers = results.get_lensed_scalar_cls(CMB_unit='muK', lmax=3000)

    # Multipole moments (l) from 0 to lmax
    ls = np.arange(powers.shape[0])

    # Extract BB spectrum (column 2) for l=2 to l=3000
    l_min = 2
    l_max = 3000
    ls_out = ls[l_min:l_max+1]
    dl_bb_out = powers[l_min:l_max+1, 2]  # D_l^{BB} in muK^2

    # Save to CSV
    output_df = pd.DataFrame({'l': ls_out, 'BB': dl_bb_out})
    output_path = os.path.join(output_dir, "result.csv")
    output_df.to_csv(output_path, index=False)

    # Print summary to console
    print("CMB B-mode polarization power spectrum (D_l^{BB} in muK^2) computed for l=2 to l=3000.")
    print("Results saved to " + output_path)
    print("First 5 rows of the output:")
    print(output_df.head())
    print("Last 5 rows of the output:")
    print(output_df.tail())

if __name__ == "__main__":
    compute_cmb_bmode_power_spectrum()