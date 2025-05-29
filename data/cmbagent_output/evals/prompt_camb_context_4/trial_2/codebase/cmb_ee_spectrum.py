# filename: codebase/cmb_ee_spectrum.py
import camb
import numpy as np
import pandas as pd
import os


def compute_cmb_ee_spectrum():
    r"""
    Compute the CMB E-mode polarization power spectrum D_l^{EE} = l(l+1)C_l^{EE}/(2pi)
    for a flat Lambda CDM cosmology using CAMB, with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (Omega_b h^2): 0.022
        - Cold dark matter density (Omega_c h^2): 0.122
        - Neutrino mass sum (Sigma m_nu): 0.06 eV
        - Curvature (Omega_k): 0
        - Optical depth to reionization (tau): 0.04
        - Scalar amplitude (A_s): 2e-9
        - Scalar spectral index (n_s): 0.965

    The spectrum is computed for multipole moments l=2 to l=3000, in units of microKelvin^2 (uK^2).
    The results are saved in 'data/result.csv' with columns:
        - l: Multipole moment (integer, 2 to 3000)
        - EE: E-mode polarization power spectrum (uK^2)
    """
    # Cosmological parameters (units in comments)
    H0 = 67.5           # Hubble constant [km/s/Mpc]
    ombh2 = 0.022       # Omega_b h^2 [dimensionless]
    omch2 = 0.122       # Omega_c h^2 [dimensionless]
    mnu = 0.06          # Sum of neutrino masses [eV]
    omk = 0.0           # Curvature [dimensionless]
    tau = 0.04          # Optical depth to reionization [dimensionless]
    As = 2.0e-9         # Scalar amplitude [dimensionless]
    ns = 0.965          # Scalar spectral index [dimensionless]
    lmax = 3000         # Maximum multipole moment [dimensionless]

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
        lens_potential_accuracy=1,
        WantScalars=True,
        WantTensors=False,
        DoLensing=True
    )

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in uK^2, D_l = l(l+1)C_l/(2pi)
    powers = results.get_cmb_power_spectra(
        params=pars,
        lmax=lmax,
        spectra=['lensed_scalar'],
        CMB_unit='muK',
        raw_cl=False
    )

    # Extract EE spectrum (column 1)
    lensed_scalar_cls = powers['lensed_scalar']  # shape: (lmax+1, 4)
    cl_EE = lensed_scalar_cls[:, 1]  # EE is column 1

    # Prepare l and EE arrays for l=2 to l=3000
    ls = np.arange(2, lmax + 1)  # l values
    cl_EE_selected = cl_EE[2:lmax + 1]  # EE spectrum for l=2..lmax

    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to CSV
    output_filename = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'l': ls, 'EE': cl_EE_selected})
    df.to_csv(output_filename, index=False)

    # Print summary to console
    print("E-mode polarization power spectrum (D_l^EE = l(l+1)C_l^EE/(2pi)) saved to data/result.csv")
    print("Units: EE in microKelvin^2 (uK^2), l is dimensionless (multipole moment)")
    print("\nFirst 5 rows:")
    with pd.option_context('display.precision', 6, 'display.max_rows', 5):
        print(df.head(5).to_string(index=False))
    print("\nLast 5 rows:")
    with pd.option_context('display.precision', 6, 'display.max_rows', 5):
        print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    compute_cmb_ee_spectrum()
