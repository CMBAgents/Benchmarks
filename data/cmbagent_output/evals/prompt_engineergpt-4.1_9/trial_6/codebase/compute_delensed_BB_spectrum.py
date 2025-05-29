# filename: codebase/compute_delensed_BB_spectrum.py
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
    tau=0.06,               # Optical depth to reionization [dimensionless]
    r=0.1,                  # Tensor-to-scalar ratio [dimensionless]
    As=2e-9,                # Scalar amplitude [dimensionless]
    ns=0.965,               # Scalar spectral index [dimensionless]
    delensing_efficiency=0.1, # Fraction of lensing B-modes removed [dimensionless]
    lmin=2,                 # Minimum multipole
    lmax=3000,              # Maximum multipole
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
    delensing_efficiency : float
        Fraction of lensing B-modes removed (e.g., 0.1 for 10%).
    lmin : int
        Minimum multipole moment (inclusive).
    lmax : int
        Maximum multipole moment (inclusive).
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
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    # 'total' includes lensing, 'unlensed_scalar' does not
    cl_total = powers['total']  # shape: (lmax+1, 4) for TT, EE, BB, TE

    # Extract BB spectrum (muK^2)
    # cl_total[:,2] is BB
    ell = np.arange(cl_total.shape[0])  # ell = 0, 1, ..., lmax
    BB = cl_total[:,2]  # muK^2

    # Apply delensing: remove delensing_efficiency fraction of lensing B-modes
    # The total BB = primordial + lensing
    # Delensed BB = primordial + (1 - delensing_efficiency) * lensing
    # Since we do not have primordial and lensing separated, approximate:
    # Delensed BB = BB - delensing_efficiency * (BB - BB_primordial)
    # But with only total BB, a common approximation is:
    # Delensed BB = (1 - delensing_efficiency) * (BB - BB_primordial) + BB_primordial
    # For this code, we use the standard approach: Delensed BB = BB - delensing_efficiency * (BB - BB_primordial)
    # But since we do not have BB_primordial, we can run CAMB again with lensing off to get it.

    # Get primordial BB (unlensed)
    powers_unlensed = powers['unlensed_total'] if 'unlensed_total' in powers else results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True, lensing=False)['total']
    BB_prim = powers_unlensed[:,2]  # muK^2

    # Compute delensed BB
    BB_delensed = BB - delensing_efficiency * (BB - BB_prim)

    # Restrict to lmin <= ell <= lmax
    mask = (ell >= lmin) & (ell <= lmax)
    ell_out = ell[mask]
    BB_out = BB_delensed[mask]

    # Save to CSV
    df = pd.DataFrame({'l': ell_out.astype(int), 'BB': BB_out})
    df.to_csv(output_csv, index=False)

    # Print summary
    print("Delensed CMB B-mode power spectrum (C_ell^{BB}) saved to " + output_csv)
    print("Columns: l (multipole moment), BB (delensed C_ell^{BB} in muK^2)")
    print("First 10 rows:")
    print(df.head(10).to_string(index=False))
    print("Last 10 rows:")
    print(df.tail(10).to_string(index=False))
    print("Total number of multipoles: " + str(len(df)))

if __name__ == "__main__":
    compute_delensed_BB_spectrum()