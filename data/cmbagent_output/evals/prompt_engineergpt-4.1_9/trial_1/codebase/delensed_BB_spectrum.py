# filename: codebase/delensed_BB_spectrum.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

def compute_delensed_BB_spectrum(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density [dimensionless]
    omch2=0.122,            # Cold dark matter density [dimensionless]
    mnu=0.06,               # Neutrino mass sum [eV]
    omk=0.0,                # Curvature [dimensionless]
    tau=0.06,               # Optical depth to reionization
    r=0.1,                  # Tensor-to-scalar ratio [dimensionless]
    As=2e-9,                # Scalar amplitude [dimensionless]
    ns=0.965,               # Scalar spectral index [dimensionless]
    lmin=2,                 # Minimum multipole
    lmax=3000,              # Maximum multipole
    delensing_efficiency=0.1, # Fractional reduction in lensing B-modes (10%)
    output_csv='data/result.csv' # Output CSV file path
):
    r"""
    Compute the delensed CMB B-mode polarization power spectrum (C_ell^{BB}) for the specified cosmology,
    apply a delensing efficiency, and save the results to a CSV file.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter.
    omch2 : float
        Physical cold dark matter density parameter.
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter.
    tau : float
        Optical depth to reionization.
    r : float
        Tensor-to-scalar ratio.
    As : float
        Scalar amplitude.
    ns : float
        Scalar spectral index.
    lmin : int
        Minimum multipole moment.
    lmax : int
        Maximum multipole moment.
    delensing_efficiency : float
        Fractional reduction in lensing B-modes (e.g., 0.1 for 10%).
    output_csv : str
        Path to output CSV file.

    Returns
    -------
    None
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=r)
    pars.WantTensors = True
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.Want_CMB_lensing = True

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=False)
    # powers is a dict with keys: 'total', 'unlensed_scalar', 'lensed_scalar', 'tensor', 'lens_potential'

    # Extract multipole moments and B-mode spectra
    ell = np.arange(powers['total'].shape[0])  # ell = 0, 1, ..., lmax
    # C_ell^{BB} in muK^2
    BB_total = powers['total'][:, 2]  # total BB (includes lensing and primordial)
    BB_unlensed = powers['unlensed_scalar'][:, 2]  # primordial BB (no lensing)
    BB_lensed = BB_total - BB_unlensed  # lensing BB

    # Apply delensing: reduce lensing BB by delensing_efficiency
    BB_delensed = BB_unlensed + (1.0 - delensing_efficiency) * BB_lensed

    # Restrict to lmin <= ell <= lmax
    mask = (ell >= lmin) & (ell <= lmax)
    ell_out = ell[mask]
    BB_out = BB_delensed[mask]

    # Save to CSV
    df = pd.DataFrame({'l': ell_out.astype(int), 'BB': BB_out})
    df.to_csv(output_csv, index=False)

    # Print summary to console
    np.set_printoptions(precision=6, suppress=True)
    pd.set_option('display.float_format', lambda x: '%.6e' + '' % x)
    print("Delensed CMB B-mode power spectrum (C_ell^{BB}) saved to " + output_csv)
    print("First 10 rows of the result (l, BB [muK^2]):")
    print(df.head(10))
    print("\nSummary statistics for BB [muK^2]:")
    print(df['BB'].describe())

if __name__ == "__main__":
    compute_delensed_BB_spectrum()