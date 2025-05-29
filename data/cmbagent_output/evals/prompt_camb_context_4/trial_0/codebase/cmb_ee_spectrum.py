# filename: codebase/cmb_ee_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_ee_spectrum():
    r"""
    Compute the lensed scalar CMB E-mode polarization power spectrum D_l^{EE} = l(l+1)C_l^{EE}/(2π)
    in units of microkelvin squared (μK^2) for a flat Lambda CDM cosmology using CAMB.

    Cosmological parameters:
        H0: Hubble constant [km/s/Mpc]
        ombh2: Baryon density Omega_b h^2
        omch2: Cold dark matter density Omega_c h^2
        mnu: Sum of neutrino masses [eV]
        omk: Curvature Omega_k
        tau: Optical depth to reionization
        As: Scalar amplitude
        ns: Scalar spectral index

    The function computes the spectrum for multipole moments l=2 to l=3000 and saves the result
    as a CSV file with columns:
        l: Multipole moment (integer, 2 to 3000)
        EE: E-mode polarization power spectrum (μK^2)

    The output file is saved as 'data/result.csv'.

    Returns
    -------
    None
    """
    # Cosmological parameters
    H0 = 67.5  # [km/s/Mpc]
    ombh2 = 0.022  # Omega_b h^2
    omch2 = 0.122  # Omega_c h^2
    mnu = 0.06  # [eV]
    omk = 0.0  # Flat universe
    tau = 0.04  # Optical depth
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    lmax = 3000
    pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=1)

    # Run CAMB
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in μK^2
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    lensed_scalar_cls = powers['lensed_scalar']  # shape: (lmax+1, 4), columns: TT, EE, BB, TE

    # Extract EE spectrum for l=2 to l=3000
    ls = np.arange(lensed_scalar_cls.shape[0])  # l=0 to lmax
    l_min = 2
    l_max = 3000
    ell_values = ls[l_min:l_max+1]
    EE_values = lensed_scalar_cls[l_min:l_max+1, 1]  # EE is column 1

    # Prepare output DataFrame
    df = pd.DataFrame({'l': ell_values.astype(int), 'EE': EE_values})

    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "result.csv")
    df.to_csv(output_path, index=False)

    # Print summary
    print("CMB E-mode polarization power spectrum (lensed scalar, μK^2) saved to data/result.csv")
    print("First five rows of the output:")
    print(df.head())
    print("Last five rows of the output:")
    print(df.tail())
    print("Total number of rows:" + str(len(df)))

if __name__ == "__main__":
    compute_cmb_ee_spectrum()
