# filename: codebase/compute_cmb_ee_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_ee_spectrum():
    r"""
    Compute the lensed scalar CMB E-mode polarization power spectrum D_l^EE = l(l+1)C_l^EE/(2pi)
    for a flat Lambda CDM cosmology using CAMB, with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (Omega_b h^2): 0.022
        - Cold dark matter density (Omega_c h^2): 0.122
        - Neutrino mass sum (Sigma m_nu): 0.06 eV
        - Curvature (Omega_k): 0
        - Optical depth to reionization (tau): 0.04
        - Scalar amplitude (A_s): 2e-9
        - Scalar spectral index (n_s): 0.965

    The spectrum is computed for multipole moments l=2 to l=3000, in units of microkelvin squared (muK^2).
    The results are saved to 'data/result.csv' with columns:
        - l: Multipole moment (integer, 2..3000)
        - EE: E-mode polarization power spectrum (muK^2)
    """
    # Set up output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Initialize CAMB parameters
    pars = camb.CAMBparams()

    # 2. Set cosmological parameters (units in docstring)
    pars.set_cosmology(
        H0=67.5,         # Hubble constant [km/s/Mpc]
        ombh2=0.022,     # Omega_b h^2 [dimensionless]
        omch2=0.122,     # Omega_c h^2 [dimensionless]
        mnu=0.06,        # Sum of neutrino masses [eV]
        omk=0,           # Curvature [dimensionless]
        tau=0.04         # Optical depth to reionization [dimensionless]
    )

    # 3. Set primordial power spectrum parameters
    pars.InitPower.set_params(
        As=2e-9,         # Scalar amplitude [dimensionless]
        ns=0.965         # Scalar spectral index [dimensionless]
    )

    # 4. Set calculation range and accuracy
    lmax = 3000
    pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=1)

    # 5. Run CAMB to get results
    results = camb.get_results(pars)

    # 6. Extract lensed scalar CMB power spectra in muK^2
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    lensed_scalar_cls = powers['lensed_scalar']  # shape: (lmax+1, 4), columns: TT, EE, BB, TE

    # 7. Extract EE spectrum for l=2..3000
    ls = np.arange(lensed_scalar_cls.shape[0])  # l values from 0 to lmax
    l_min = 2
    l_max = 3000
    ell_values = ls[l_min:l_max+1]              # l=2..3000
    EE_values = lensed_scalar_cls[l_min:l_max+1, 1]  # EE column (index 1), units: muK^2

    # 8. Save to CSV
    output_df = pd.DataFrame({
        'l': ell_values.astype(int),
        'EE': EE_values
    })
    output_path = os.path.join(output_dir, "result.csv")
    output_df.to_csv(output_path, index=False)

    # Print summary to console
    print("CMB E-mode polarization power spectrum (lensed scalar, muK^2) saved to data/result.csv")
    print("First five rows of the output:")
    print(output_df.head())
    print("Last five rows of the output:")
    print(output_df.tail())

if __name__ == "__main__":
    compute_cmb_ee_spectrum()