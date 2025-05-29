# filename: codebase/compute_delensed_BB_spectrum.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

def compute_delensed_BB_spectrum(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density Omega_b h^2
    omch2=0.122,            # Cold dark matter density Omega_c h^2
    mnu=0.06,               # Neutrino mass sum [eV]
    omk=0.0,                # Curvature Omega_k
    tau=0.06,               # Optical depth to reionization
    r=0.1,                  # Tensor-to-scalar ratio
    As=2e-9,                # Scalar amplitude
    ns=0.965,               # Scalar spectral index
    lmin=2,                 # Minimum multipole
    lmax=3000,              # Maximum multipole
    delensing_efficiency=0.1, # Fraction of lensing B-modes removed (10%)
    output_csv_path="data/result.csv"
):
    r"""
    Compute the delensed CMB B-mode polarization power spectrum (C_ell^{BB}) using CAMB.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Baryon density Omega_b h^2.
    omch2 : float
        Cold dark matter density Omega_c h^2.
    mnu : float
        Neutrino mass sum in eV.
    omk : float
        Curvature Omega_k.
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
        Fraction of lensing B-modes removed (e.g., 0.1 for 10%).
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
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.WantTensors = True
    pars.Want_CMB_lensing = True

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    # powers is a dict with keys: 'total', 'unlensed_scalar', 'lensed_scalar', 'tensor', 'lens_potential'

    # Extract relevant spectra
    lensed = powers['lensed_scalar']  # shape: (lmax+1, 4) [TT, EE, BB, TE]
    tensor = powers['tensor']         # shape: (lmax+1, 4)
    unlensed = powers['unlensed_scalar']  # shape: (lmax+1, 4)

    # l array: multipole moments
    ell = np.arange(lensed.shape[0])  # 0 to lmax

    # B-mode: index 2
    BB_lensed = lensed[:,2]   # lensed BB (includes lensing + primordial)
    BB_tensor = tensor[:,2]   # primordial BB (from tensors only)
    BB_unlensed = unlensed[:,2] # scalar-only BB (should be ~0)

    # Lensing B-modes = lensed BB - primordial BB
    BB_lensing = BB_lensed - BB_tensor

    # Apply delensing: remove delensing_efficiency fraction of lensing B-modes
    BB_delensed = BB_tensor + (1.0 - delensing_efficiency) * BB_lensing

    # Restrict to lmin <= l <= lmax
    mask = (ell >= lmin) & (ell <= lmax)
    ell_out = ell[mask]
    BB_out = BB_delensed[mask]  # Already in muK^2

    # Save to CSV
    df = pd.DataFrame({'l': ell_out.astype(int), 'BB': BB_out})
    df.to_csv(output_csv_path, index=False)

    # Print summary
    print("Delensed CMB B-mode power spectrum (C_ell^{BB}) computed and saved to " + output_csv_path)
    print("Columns: l (multipole moment), BB (delensed C_ell^{BB} in micro-Kelvin^2)")
    print("First 10 rows:")
    print(df.head(10).to_string(index=False))
    print("\nLast 10 rows:")
    print(df.tail(10).to_string(index=False))
    print("\nTotal number of multipoles saved: " + str(len(df)))

if __name__ == "__main__":
    compute_delensed_BB_spectrum()