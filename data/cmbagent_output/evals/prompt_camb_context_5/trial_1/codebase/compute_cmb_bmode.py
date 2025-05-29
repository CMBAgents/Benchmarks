# filename: codebase/compute_cmb_bmode.py
import camb
import numpy as np
import pandas as pd
import os


def compute_cmb_bmode_power_spectrum():
    r"""
    Compute the CMB B-mode polarization power spectrum D_l^{BB} = l(l+1)C_l^{BB}/(2\pi)
    for a flat Lambda CDM cosmology using CAMB, with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Neutrino mass sum (mnu): 0.06 eV
        - Curvature (omk): 0
        - Optical depth to reionization (tau): 0.06
        - Tensor-to-scalar ratio (r): 0
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965
    The B-mode power spectrum is computed in units of micro-Kelvin squared (uK^2)
    for multipole moments l from 2 to 3000. The results are saved in a CSV file
    'result.csv' in the 'data/' directory, with columns:
        - l: Multipole moment (integer, 2 to 3000)
        - BB: B-mode polarization power spectrum (uK^2)
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cosmological parameters (units in comments)
    H0 = 67.5                # Hubble constant [km/s/Mpc]
    ombh2 = 0.022            # Baryon density
    omch2 = 0.122            # Cold dark matter density
    mnu = 0.06               # Neutrino mass sum [eV]
    omk = 0.0                # Curvature
    tau = 0.06               # Optical depth to reionization
    r_tensor = 0.0           # Tensor-to-scalar ratio
    As = 2e-9                # Scalar amplitude
    ns = 0.965               # Scalar spectral index
    lmax = 3000              # Maximum multipole moment

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
        r=r_tensor,
        lmax=lmax,
        lens_potential_accuracy=1
    )

    # Run CAMB and get results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in muK^2
    # Output: array of shape (lmax+1, 4), columns: TT, EE, BB, TE
    cl = results.get_lensed_scalar_cls(CMB_unit='muK', lmax=lmax)

    # Multipole moments (l = 0 to lmax)
    ls = np.arange(cl.shape[0])

    # Extract BB spectrum (column 2)
    BB = cl[:, 2]

    # Select l = 2 to 3000
    mask = (ls >= 2) & (ls <= 3000)
    ls_out = ls[mask]
    BB_out = BB[mask]

    # Prepare DataFrame and save to CSV
    df = pd.DataFrame({'l': ls_out, 'BB': BB_out})
    output_path = os.path.join(output_dir, "result.csv")
    df.to_csv(output_path, index=False)

    # Print summary to console
    print("CMB B-mode polarization power spectrum (lensed, scalar, muK^2) computed for l=2 to 3000.")
    print("Results saved to " + output_path)
    print("First 5 rows of the output:")
    print(df.head())
    print("Last 5 rows of the output:")
    print(df.tail())
    print("Total number of rows: " + str(len(df)))


if __name__ == "__main__":
    compute_cmb_bmode_power_spectrum()