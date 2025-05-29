# filename: codebase/cmb_bmode_power_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_bmode_power_spectrum():
    r"""
    Compute the CMB B-mode polarization power spectrum D_l^{BB} = l(l+1)C_l^{BB}/(2\pi)
    for a flat Lambda CDM cosmology using CAMB, with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (Omega_b h^2): 0.022
        - Cold dark matter density (Omega_c h^2): 0.122
        - Neutrino mass sum (Sigma m_nu): 0.06 eV
        - Curvature (Omega_k): 0
        - Optical depth to reionization (tau): 0.06
        - Tensor-to-scalar ratio (r): 0
        - Scalar amplitude (A_s): 2e-9
        - Scalar spectral index (n_s): 0.965

    The B-mode power spectrum is computed in units of micro-Kelvin squared (muK^2)
    for multipole moments l from 2 to 3000. The results are saved in a CSV file
    'result.csv' in the 'data/' directory, with columns:
        - l: Multipole moment (integer, 2 to 3000)
        - BB: B-mode polarization power spectrum (muK^2)
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cosmological parameters (units in comments)
    H0 = 67.5                # Hubble constant [km/s/Mpc]
    ombh2 = 0.022            # Baryon density [dimensionless]
    omch2 = 0.122            # Cold dark matter density [dimensionless]
    mnu = 0.06               # Neutrino mass sum [eV]
    omk = 0.0                # Curvature [dimensionless]
    tau = 0.06               # Optical depth to reionization [dimensionless]
    r_tensor = 0.0           # Tensor-to-scalar ratio [dimensionless]
    As = 2e-9                # Scalar amplitude [dimensionless]
    ns = 0.965               # Scalar spectral index [dimensionless]
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

    # Get lensed scalar CMB power spectra in muK^2 units
    # Returns array of shape (lmax+1, 4): columns are TT, EE, BB, TE
    cls = results.get_lensed_scalar_cls(CMB_unit='muK', lmax=lmax)

    # Multipole moments (l) and B-mode power spectrum (BB)
    ls = np.arange(cls.shape[0])  # l = 0 ... lmax
    BB = cls[:, 2]                # BB spectrum (muK^2)

    # Select l = 2 to 3000
    lmin = 2
    lmax_out = 3000
    mask = (ls >= lmin) & (ls <= lmax_out)
    ls_out = ls[mask]
    BB_out = BB[mask]

    # Save to CSV
    output_df = pd.DataFrame({'l': ls_out, 'BB': BB_out})
    output_path = os.path.join(output_dir, "result.csv")
    output_df.to_csv(output_path, index=False)

    # Print summary to console
    print("CMB B-mode polarization power spectrum (lensed, scalar, muK^2) saved to " + output_path)
    print("First 10 rows of the output:")
    print(output_df.head(10))
    print("\nColumns: l (multipole moment), BB (B-mode power spectrum in muK^2)")


if __name__ == "__main__":
    compute_cmb_bmode_power_spectrum()