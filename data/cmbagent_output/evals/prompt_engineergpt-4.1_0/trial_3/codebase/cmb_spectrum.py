# filename: codebase/cmb_spectrum.py
r"""
Compute the CMB temperature power spectrum for a flat Lambda CDM cosmology using CAMB.

Parameters:
    H0 (float): Hubble constant in km/s/Mpc.
    ombh2 (float): Baryon density parameter Omega_b h^2.
    omch2 (float): Cold dark matter density parameter Omega_c h^2.
    mnu (float): Sum of neutrino masses in eV.
    omk (float): Curvature parameter Omega_k.
    tau (float): Optical depth to reionization.
    As (float): Scalar amplitude.
    ns (float): Scalar spectral index.

Outputs:
    - CSV file 'data/result.csv' with columns:
        l: Multipole moment (integer, 2 to 3000)
        TT: Temperature power spectrum l(l+1)C_l^{TT}/(2pi) in microK^2

All units are SI except where specified.
"""

import os
import numpy as np
import pandas as pd

# Ensure output directory exists
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    import camb
    from camb import model, initialpower
except ImportError as e:
    print("CAMB package is required but not installed. Please install camb and rerun the code.")
    raise e


def compute_cmb_tt_spectrum(
    H0=67.5,           # Hubble constant [km/s/Mpc]
    ombh2=0.02,        # Omega_b h^2 [dimensionless]
    omch2=0.122,       # Omega_c h^2 [dimensionless]
    mnu=0.06,          # Sum of neutrino masses [eV]
    omk=0.0,           # Curvature [dimensionless]
    tau=0.06,          # Optical depth [dimensionless]
    As=2e-9,           # Scalar amplitude [dimensionless]
    ns=0.965,          # Scalar spectral index [dimensionless]
    lmin=2,            # Minimum multipole
    lmax=3000          # Maximum multipole
):
    r"""
    Compute the CMB temperature power spectrum l(l+1)C_l^{TT}/(2pi) in microK^2.

    Returns:
        l (ndarray): Multipole moments (lmin to lmax)
        TT (ndarray): Temperature power spectrum in microK^2
    """
    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.WantTensors = False

    # Calculate results for these parameters
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total'])
    totCL = powers['total']

    # totCL shape: (lmax+1, 4) columns: TT, EE, BB, TE
    # l values: 0 to lmax
    ell = np.arange(totCL.shape[0])  # l = 0 ... lmax

    # Select l in [lmin, lmax]
    mask = (ell >= lmin) & (ell <= lmax)
    ell = ell[mask]
    cltt = totCL[mask, 0]  # TT spectrum in muK^2

    # Compute l(l+1)C_l/(2pi)
    tt_spectrum = ell * (ell + 1) * cltt / (2.0 * np.pi)  # [muK^2]

    return ell, tt_spectrum


# Compute the spectrum
l, TT = compute_cmb_tt_spectrum()

# Save to CSV
df = pd.DataFrame({'l': l, 'TT': TT})
csv_path = os.path.join(output_dir, "result.csv")
df.to_csv(csv_path, index=False)

# Print summary of results
np.set_printoptions(precision=6, suppress=True)
print("CMB TT power spectrum computed for l = 2 to 3000.")
print("Results saved to " + csv_path)
print("First 5 rows:")
print(df.head(5).to_string(index=False))
print("Last 5 rows:")
print(df.tail(5).to_string(index=False))
