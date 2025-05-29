# filename: codebase/camb_delensed_BB.py
r"""
Compute the delensed CMB B-mode polarization power spectrum (C_ell^{BB}) using CAMB for a flat Lambda CDM cosmology,
apply a 10% delensing efficiency, and save the results to a CSV file.

Units:
- l: dimensionless multipole moment (integer)
- BB: C_ell^{BB} in microkelvin^2 (uK^2)
"""

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
    output_csv="data/result.csv"
):
    r"""
    Compute the delensed CMB B-mode power spectrum and save to CSV.

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
        Minimum multipole moment.
    lmax : int
        Maximum multipole moment.
    delensing_efficiency : float
        Fraction of lensing B-modes removed (0.1 = 10%).
    output_csv : str
        Path to output CSV file.
    """
    # Ensure output directory exists
    outdir = os.path.dirname(output_csv)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)

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
    # 'total' includes lensed scalar + tensor + lensing
    # 'unlensed_scalar' is scalar only, no lensing
    # 'tensor' is tensor only, no lensing
    # 'lensed_scalar' is scalar + lensing, no tensor

    # Extract lensed BB (total) and unlensed BB (tensor)
    cl_total = powers['total']  # shape: (lmax+1, 4) [TT, EE, BB, TE]
    cl_tensor = powers['tensor']  # shape: (lmax+1, 4)

    # l values
    ell = np.arange(cl_total.shape[0])  # 0 ... lmax

    # Get BB spectra
    BB_total = cl_total[:,2]  # lensed BB (includes lensing + tensor)
    BB_tensor = cl_tensor[:,2]  # primordial tensor BB (no lensing)

    # Lensing BB = lensed BB - primordial tensor BB
    BB_lensing = BB_total - BB_tensor

    # Delensed BB = primordial tensor BB + (1 - delensing_efficiency) * lensing BB
    BB_delensed = BB_tensor + (1.0 - delensing_efficiency) * BB_lensing

    # Restrict to lmin <= l <= lmax
    mask = (ell >= lmin) & (ell <= lmax)
    ell_out = ell[mask]
    BB_out = BB_delensed[mask]

    # Save to CSV
    df = pd.DataFrame({'l': ell_out, 'BB': BB_out})
    df.to_csv(output_csv, index=False)

    # Print summary
    print("Delensed CMB B-mode power spectrum (C_ell^{BB}) computed and saved to " + output_csv)
    print("Columns:")
    print("  l  : Multipole moment (dimensionless)")
    print("  BB : Delensed B-mode power spectrum (microkelvin^2)")
    print("First 10 rows:")
    print(df.head(10).to_string(index=False))
    print("Last 10 rows:")
    print(df.tail(10).to_string(index=False))
    print("Total number of multipoles saved: " + str(len(df)))

if __name__ == "__main__":
    compute_delensed_BB_spectrum()
