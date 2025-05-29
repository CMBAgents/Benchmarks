# filename: codebase/camb_delensed_spectrum.py
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
    omk=0,                  # Curvature [dimensionless]
    tau=0.06,               # Optical depth to reionization [dimensionless]
    r=0.1,                  # Tensor-to-scalar ratio [dimensionless]
    As=2e-9,                # Scalar amplitude [dimensionless]
    ns=0.965,               # Scalar spectral index [dimensionless]
    lmin=2,                 # Minimum multipole
    lmax=3000,              # Maximum multipole
    delensing_efficiency=0.1, # Fraction of lensing B-modes removed
    output_csv='data/result.csv'
):
    r"""
    Compute the delensed CMB B-mode polarization power spectrum (C_ell^{BB}) for a flat Lambda CDM cosmology
    using CAMB, and save the results to a CSV file.

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
        Fraction of lensing B-modes removed (e.g., 0.1 for 10%).
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
    pars.Want_CMB_lensing = True
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.set_accuracy(AccuracyBoost=2.0, lSampleBoost=2.0, lAccuracyBoost=2.0)

    # Get results from CAMB
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    # powers is a dict with keys: 'total', 'unlensed_scalar', 'lensed_scalar', 'tensor', 'lens_potential'

    # Extract relevant spectra
    cl_total = powers['total']      # includes lensing and tensors
    cl_unlensed = powers['unlensed_scalar']  # scalar only, no lensing
    cl_tensor = powers['tensor']    # tensor only, no lensing
    cl_lensed = powers['lensed_scalar']  # scalar + lensing, no tensors

    # Multipole array
    ell = np.arange(cl_total.shape[0])  # ell[0] = 0

    # B-mode spectra
    # cl_total[:, 2] = BB (includes lensing + tensor)
    # cl_lensed[:, 2] = BB (lensing only, no tensor)
    # cl_tensor[:, 2] = BB (tensor only, no lensing)
    # cl_unlensed[:, 2] = BB (scalar only, should be zero)

    # Calculate lensing B-modes: lensing_BB = lensed_scalar_BB - unlensed_scalar_BB
    lensing_BB = cl_lensed[:, 2] - cl_unlensed[:, 2]  # [muK^2]
    tensor_BB = cl_tensor[:, 2]                       # [muK^2]

    # Delensed BB: tensor + (1 - delensing_efficiency) * lensing
    delensed_BB = tensor_BB + (1.0 - delensing_efficiency) * lensing_BB

    # Restrict to lmin <= ell <= lmax
    mask = (ell >= lmin) & (ell <= lmax)
    ell_out = ell[mask]
    BB_out = delensed_BB[mask]

    # Save to CSV
    df = pd.DataFrame({'l': ell_out, 'BB': BB_out})
    df.to_csv(output_csv, index=False)

    # Print summary
    print("Delensed CMB B-mode power spectrum (C_ell^{BB}) saved to " + output_csv)
    print("Columns: l (multipole moment), BB (delensed C_ell^{BB} in microK^2)")
    print("First 10 rows:")
    print(df.head(10).to_string(index=False))
    print("\nSummary statistics for BB (microK^2):")
    print(df['BB'].describe())

if __name__ == "__main__":
    compute_delensed_BB_spectrum()