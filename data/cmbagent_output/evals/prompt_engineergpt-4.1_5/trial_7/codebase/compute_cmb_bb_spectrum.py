# filename: codebase/compute_cmb_bb_spectrum.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

def compute_cmb_bb_spectrum():
    r"""
    Computes the CMB B-mode polarization power spectrum (l(l+1)C_l^{BB}/(2pi)) for a flat Lambda CDM cosmology
    using CAMB, with specified cosmological parameters.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns:
        - l: Multipole moment (integer, 2 <= l <= 3000)
        - BB: B-mode polarization power spectrum (l(l+1)C_l^{BB}/(2pi)) in muK^2
    """
    # Set cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    r_tensor = 0.0  # Tensor-to-scalar ratio [dimensionless]
    As = 2e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=r_tensor)
    pars.set_for_lmax(3000, lens_potential_accuracy=1)
    pars.WantTensors = True  # Needed for BB spectrum, even if r=0

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000)
    # 'total' includes lensing, 'unlensed_scalar' does not
    # For r=0, BB is only from lensing (if lensing is on)
    BB = powers['total'][:, 2]  # BB is column 2

    # l values: powers['total'] is indexed from l=0, so l = np.arange(powers['total'].shape[0])
    lmax = 3000
    l = np.arange(powers['total'].shape[0])
    # Only keep l=2 to l=3000
    lmin = 2
    l_mask = (l >= lmin) & (l <= lmax)
    l = l[l_mask]
    BB = BB[l_mask]

    # Compute l(l+1)C_l^{BB}/(2pi)
    # BB is C_l^{BB} in muK^2
    BB_power = l * (l + 1) * BB / (2.0 * np.pi)  # [muK^2]

    # Save to CSV
    if not os.path.exists("data"):
        os.makedirs("data")
    df = pd.DataFrame({'l': l, 'BB': BB_power})
    df.to_csv("data/result.csv", index=False)

    # Print summary
    pd.set_option("display.precision", 6)
    pd.set_option("display.max_rows", 10)
    print("CMB B-mode polarization power spectrum (l(l+1)C_l^{BB}/(2pi)) [muK^2] for l=2 to l=3000:")
    print(df.head(10))
    print("...")
    print(df.tail(10))
    print("\nSaved full results to data/result.csv")

    return df


if __name__ == "__main__":
    compute_cmb_bb_spectrum()
