# filename: codebase/delensed_BB_spectrum.py
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
    omk=0,                  # Curvature [Omega_k]
    tau=0.06,               # Optical depth to reionization
    r=0.1,                  # Tensor-to-scalar ratio
    As=2e-9,                # Scalar amplitude
    ns=0.965,               # Scalar spectral index
    lmin=2,                 # Minimum multipole
    lmax=3000,              # Maximum multipole
    delensing_efficiency=0.1, # Fraction of lensing B-modes removed (10%)
    output_dir="data/",
    output_filename="result.csv"
):
    r"""
    Compute the delensed CMB B-mode polarization power spectrum (C_ell^{BB}) for a flat Lambda CDM cosmology
    using CAMB, with specified cosmological parameters and delensing efficiency.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Baryon density parameter Omega_b h^2.
    omch2 : float
        Cold dark matter density parameter Omega_c h^2.
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter Omega_k.
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
    output_dir : str
        Directory to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
        Saves the delensed B-mode power spectrum to a CSV file and prints a summary.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
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

    # Extract lensed and unlensed BB spectra
    lensed = powers['total']      # includes lensing and tensors
    unlensed = powers['unlensed_scalar']  # scalar only, no lensing, no tensors
    tensor = powers['tensor']     # tensor only, no lensing

    # Get multipole array
    ell = np.arange(lensed.shape[0])  # ell[0] = 0, ell[1] = 1, ..., up to lmax

    # The BB spectrum is at column 2 (TT, EE, BB, TE)
    BB_lensed = lensed[:,2]      # [muK^2]
    BB_unlensed = unlensed[:,2]  # [muK^2]
    BB_tensor = tensor[:,2]      # [muK^2]

    # Lensing B-modes = lensed - unlensed - tensor
    BB_lensing = BB_lensed - BB_unlensed - BB_tensor

    # Delensed BB: remove a fraction of lensing B-modes
    BB_delensed = BB_lensed - delensing_efficiency * BB_lensing

    # Restrict to lmin <= ell <= lmax
    mask = (ell >= lmin) & (ell <= lmax)
    ell_out = ell[mask]
    BB_out = BB_delensed[mask]

    # Save to CSV
    df = pd.DataFrame({'l': ell_out, 'BB': BB_out})
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)

    # Print summary
    print("Delensed CMB B-mode power spectrum (C_ell^{BB}) saved to " + output_path)
    print("Columns: l (multipole moment), BB (delensed C_ell^{BB} in microK^2)")
    print("First 10 rows:")
    print(df.head(10).to_string(index=False))
    print("\nSummary statistics for BB (microK^2):")
    print(df['BB'].describe())

if __name__ == "__main__":
    compute_delensed_BB_spectrum()