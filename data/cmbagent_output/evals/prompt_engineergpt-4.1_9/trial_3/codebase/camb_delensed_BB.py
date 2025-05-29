# filename: codebase/camb_delensed_BB.py
import numpy as np
import camb
from camb import model, initialpower
import pandas as pd
import os

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
    output_csv='data/result.csv'
):
    r"""
    Compute the delensed CMB B-mode polarization power spectrum (C_ell^{BB}) for a flat Lambda CDM cosmology
    using CAMB, applying a specified delensing efficiency.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter (Omega_b h^2).
    omch2 : float
        Physical cold dark matter density parameter (Omega_c h^2).
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
        Fraction of lensing B-modes to remove (0.1 = 10%).
    output_csv : str
        Path to output CSV file.

    Returns
    -------
    None
        Saves the delensed B-mode power spectrum to a CSV file and prints a summary.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=r)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.WantTensors = True
    pars.Want_CMB_lensing = True

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total', 'unlensed_scalar', 'lensed_scalar', 'tensor', 'lensed_total'])

    # Get lensed total BB spectrum (includes both primordial and lensing B-modes)
    BB_lensed = powers['lensed_total'][:, 2]  # Index 2 is BB
    ells = np.arange(BB_lensed.size)

    # Get unlensed tensor BB spectrum (primordial B-modes only)
    BB_tensor = powers['tensor'][:, 2]

    # Compute lensing B-modes: BB_lens = BB_lensed - BB_tensor
    BB_lens = BB_lensed - BB_tensor

    # Apply delensing: remove a fraction of lensing B-modes
    BB_delensed = BB_tensor + (1.0 - delensing_efficiency) * BB_lens

    # Restrict to requested multipole range
    mask = (ells >= lmin) & (ells <= lmax)
    ells_out = ells[mask]
    BB_out = BB_delensed[mask]

    # Save to CSV
    df = pd.DataFrame({'l': ells_out.astype(int), 'BB': BB_out})
    df.to_csv(output_csv, index=False)

    # Print summary
    print("Delensed CMB B-mode power spectrum (C_ell^{BB}) saved to " + output_csv)
    print("Columns: l (multipole), BB (delensed C_ell^{BB} in microK^2)")
    print("Multipole range: " + str(lmin) + " <= l <= " + str(lmax))
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())
    print("Delensing efficiency applied: " + str(delensing_efficiency * 100.0) + " % (fraction of lensing B-modes removed)")

if __name__ == "__main__":
    compute_delensed_BB_spectrum()
