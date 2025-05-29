# filename: codebase/compute_delensed_BB_spectrum.py
import numpy as np
import pandas as pd
import os
from camb import model, initialpower, get_results, get_power_spectra, CAMBparams

def compute_delensed_BB_spectrum(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density [dimensionless]
    omch2=0.122,            # Cold dark matter density [dimensionless]
    mnu=0.06,               # Neutrino mass sum [eV]
    omk=0.0,                # Curvature [dimensionless]
    tau=0.06,               # Optical depth to reionization [dimensionless]
    r=0.1,                  # Tensor-to-scalar ratio [dimensionless]
    As=2e-9,                # Scalar amplitude [dimensionless]
    ns=0.965,               # Scalar spectral index [dimensionless]
    lmin=2,                 # Minimum multipole
    lmax=3000,              # Maximum multipole
    delensing_efficiency=0.1, # Fraction of lensing B-modes removed (10%)
    output_csv_path="data/result.csv"
):
    r"""
    Compute the delensed CMB B-mode polarization power spectrum (C_ell^{BB}) for a flat Lambda CDM cosmology.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Baryon density parameter (Omega_b h^2).
    omch2 : float
        Cold dark matter density parameter (Omega_c h^2).
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter (Omega_k).
    tau : float
        Optical depth to reionization.
    r : float
        Tensor-to-scalar ratio.
    As : float
        Scalar amplitude.
    ns : float
        Scalar spectral index.
    lmin : int
        Minimum multipole moment (inclusive).
    lmax : int
        Maximum multipole moment (inclusive).
    delensing_efficiency : float
        Fraction of lensing B-modes removed (e.g., 0.1 for 10%).
    output_csv_path : str
        Path to save the resulting CSV file.

    Returns
    -------
    None
        Saves the delensed B-mode power spectrum to a CSV file.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up CAMB parameters
    pars = CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=r)
    pars.WantTensors = True
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.Want_CMB_lensing = True

    # Calculate results
    results = get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=False)
    # CMB_unit='muK' gives output in microkelvin^2

    # Extract lensed and unlensed spectra
    lensed = powers['lensed_scalar']
    unlensed = powers['unlensed_scalar']
    lensing = powers['lensing']

    # lensed[:, 2] is BB (B-mode) spectrum
    # lensing[:, 2] is lensing B-mode only (no primordial)
    # unlensed[:, 2] is primordial BB only (no lensing)

    # Get ell array
    ell = np.arange(lensed.shape[0])  # ell = 0, 1, ..., lmax

    # Only keep ell >= lmin
    mask = (ell >= lmin) & (ell <= lmax)
    ell = ell[mask]

    # Primordial BB (unlensed), lensing BB, total lensed BB
    BB_prim = unlensed[:, 2][mask]    # [muK^2]
    BB_lens = lensing[:, 2][mask]     # [muK^2]
    BB_total = lensed[:, 2][mask]     # [muK^2]

    # Delensed BB: remove a fraction of lensing BB
    BB_delensed = BB_prim + (1.0 - delensing_efficiency) * BB_lens

    # Save to CSV
    df = pd.DataFrame({'l': ell.astype(int), 'BB': BB_delensed})
    df.to_csv(output_csv_path, index=False)

    # Print summary
    pd.set_option("display.precision", 6)
    pd.set_option("display.max_rows", 10)
    print("Delensed CMB B-mode power spectrum (C_ell^{BB}) saved to " + output_csv_path)
    print("Columns: l (multipole), BB (delensed C_ell^{BB} in microkelvin^2)")
    print("First and last few rows of the result:")
    print(pd.concat([df.head(5), df.tail(5)]))

if __name__ == "__main__":
    compute_delensed_BB_spectrum()