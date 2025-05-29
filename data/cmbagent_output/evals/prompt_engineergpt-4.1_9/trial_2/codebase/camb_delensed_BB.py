# filename: codebase/camb_delensed_BB.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

def compute_delensed_BB_spectrum(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density [Omega_b h^2]
    omch2=0.122,            # Cold dark matter density [Omega_c h^2]
    mnu=0.06,               # Neutrino mass sum [eV]
    omk=0.0,                # Curvature [Omega_k]
    tau=0.06,               # Optical depth to reionization
    r=0.1,                  # Tensor-to-scalar ratio
    As=2e-9,                # Scalar amplitude
    ns=0.965,               # Scalar spectral index
    lmin=2,                 # Minimum multipole
    lmax=3000,              # Maximum multipole
    delensing_efficiency=0.10, # Fractional reduction in lensing B-modes (10%)
    output_csv_path="data/result.csv"
):
    r"""
    Compute the delensed CMB B-mode polarization power spectrum (C_ell^{BB}) using CAMB.

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
        Fractional reduction in lensing B-modes (e.g., 0.10 for 10%).
    output_csv_path : str
        Path to save the output CSV file.

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
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=r)
    pars.WantTensors = True
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.Want_CMB_lensing = True

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=False)
    # powers is a dict with keys: 'total', 'unlensed_scalar', 'lensed_scalar', 'tensor', 'lensed_total'

    # Extract lensed and unlensed BB spectra
    lensed = powers['lensed_total']  # shape: (lmax+1, 4) [TT, EE, BB, TE]
    unlensed = powers['unlensed_scalar']  # shape: (lmax+1, 4)
    tensor = powers['tensor']  # shape: (lmax+1, 4)

    # l values: 0 to lmax
    ells = np.arange(lensed.shape[0])

    # Get BB spectra
    BB_lensed = lensed[:,2]   # [muK^2]
    BB_unlensed = unlensed[:,2]  # [muK^2]
    BB_tensor = tensor[:,2]   # [muK^2]

    # Lensing B-modes = lensed - unlensed
    BB_lensing = BB_lensed - BB_unlensed

    # Apply delensing: reduce lensing B-modes by delensing_efficiency
    BB_delensed = BB_unlensed + (1.0 - delensing_efficiency) * BB_lensing

    # Restrict to lmin <= l <= lmax
    l_range = np.arange(lmin, lmax+1)
    BB_delensed_range = BB_delensed[lmin:lmax+1]

    # Save to CSV
    df = pd.DataFrame({'l': l_range, 'BB': BB_delensed_range})
    df.to_csv(output_csv_path, index=False)

    # Print summary
    print("Delensed CMB B-mode power spectrum (C_ell^{BB}) computed and saved to " + output_csv_path)
    print("Columns: l (multipole moment), BB (delensed C_ell^{BB} in micro-Kelvin^2)")
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())
    print("Total number of multipoles: " + str(len(df)))


if __name__ == "__main__":
    compute_delensed_BB_spectrum()