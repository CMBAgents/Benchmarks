# filename: codebase/compute_cmb_ee.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_ee_spectrum():
    r"""
    Compute the CMB E-mode polarization power spectrum D_l^{EE} = l(l+1)C_l^{EE}/(2\pi)
    for a flat Lambda CDM cosmology using CAMB, with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Neutrino mass sum (mnu): 0.06 eV
        - Curvature (omk): 0
        - Optical depth to reionization (tau): 0.04
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965
    The spectrum is computed for multipole moments l=2 to l=3000, in units of microKelvin^2.
    The results are saved to 'data/result.csv' with columns:
        - l: Multipole moment (integer, 2 to 3000)
        - EE: E-mode polarization power spectrum (microKelvin^2)
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cosmological parameters
    H0 = 67.5  # [km/s/Mpc]
    ombh2 = 0.022  # [dimensionless]
    omch2 = 0.122  # [dimensionless]
    mnu = 0.06  # [eV]
    omk = 0.0  # [dimensionless]
    tau = 0.04  # [dimensionless]
    As = 2.0e-9  # [dimensionless]
    ns = 0.965  # [dimensionless]
    lmax = 3000  # [dimensionless]

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

    # Run CAMB
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in microKelvin^2 units
    powers = results.get_cmb_power_spectra(
        params=pars,
        lmax=lmax,
        spectra=['lensed_scalar'],
        CMB_unit='muK',
        raw_cl=False
    )

    # Extract EE spectrum (column 1) for l=2 to l=3000
    lensed_scalar_cls = powers['lensed_scalar']  # shape: (lmax+1, 4)
    ls = np.arange(2, lmax + 1)  # l values from 2 to 3000
    cl_EE = lensed_scalar_cls[2:lmax + 1, 1]  # EE spectrum in microKelvin^2

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'l': ls, 'EE': cl_EE})
    df.to_csv(output_path, index=False)

    # Print summary of results
    np.set_printoptions(precision=6, suppress=True)
    print("E-mode polarization power spectrum (D_l^EE = l(l+1)C_l^EE/(2pi)) saved to " + output_path)
    print("Units: microKelvin^2")
    print("First 5 rows:")
    print(df.head(5).to_string(index=False))
    print("Last 5 rows:")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    compute_cmb_ee_spectrum()