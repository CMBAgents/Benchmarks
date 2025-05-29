# filename: codebase/camb_cmb_power_spectrum.py
r"""
Compute the CMB temperature power spectrum for a non-flat Lambda CDM cosmology using CAMB.

Parameters:
    H0: Hubble constant [km/s/Mpc]
    ombh2: Baryon density parameter [dimensionless]
    omch2: Cold dark matter density parameter [dimensionless]
    mnu: Sum of neutrino masses [eV]
    omk: Curvature parameter [dimensionless]
    tau: Optical depth to reionization [dimensionless]
    As: Scalar amplitude [dimensionless]
    ns: Scalar spectral index [dimensionless]

Output:
    Saves a CSV file 'data/result.csv' with columns:
        l: Multipole moment (integer, 2 to 3000)
        TT: Temperature power spectrum l(l+1)C_l^{TT}/(2pi) [microK^2]
    Prints a summary of the results to the console.
"""

import os
import numpy as np
import pandas as pd

# Ensure data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

try:
    import camb
    from camb import model, initialpower
except ImportError as e:
    print("CAMB is not installed. Please install camb and rerun the code.")
    raise e

# Cosmological parameters
H0 = 67.3  # Hubble constant [km/s/Mpc]
ombh2 = 0.022  # Omega_b h^2 [dimensionless]
omch2 = 0.122  # Omega_c h^2 [dimensionless]
mnu = 0.06  # Sum of neutrino masses [eV]
omk = 0.05  # Curvature [dimensionless]
tau = 0.06  # Optical depth [dimensionless]
As = 2e-9  # Scalar amplitude [dimensionless]
ns = 0.965  # Scalar spectral index [dimensionless]

# Set up CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns)
pars.set_for_lmax(3000, lens_potential_accuracy=0)
pars.WantTensors = False
pars.Want_CMB = True

# Calculate results
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')

# Extract the total TT spectrum
totCL = powers['total']
# totCL shape: (lmax+1, 4), columns: TT, EE, BB, TE

# Multipole moments
lmax = 3000
ls = np.arange(totCL.shape[0])  # l = 0 ... lmax

# Compute l(l+1)C_l^{TT}/(2pi) in microK^2
TT = totCL[:, 0]  # TT spectrum [muK^2]
cl_TT = ls * (ls + 1) * TT / (2 * np.pi)  # [muK^2]

# Select l=2 to l=3000
lmin = 2
lmax = 3000
mask = (ls >= lmin) & (ls <= lmax)
ls_out = ls[mask]
cl_TT_out = cl_TT[mask]

# Save to CSV
df = pd.DataFrame({'l': ls_out, 'TT': cl_TT_out})
csv_path = os.path.join("data", "result.csv")
df.to_csv(csv_path, index=False)

# Print summary
np.set_printoptions(precision=4, suppress=True, linewidth=120)
print("CMB temperature power spectrum (l(l+1)C_l^{TT}/(2pi)) [microK^2] for non-flat Lambda CDM cosmology")
print("Parameters:")
print("  H0 = " + str(H0) + " km/s/Mpc")
print("  Omega_b h^2 = " + str(ombh2))
print("  Omega_c h^2 = " + str(omch2))
print("  Sum m_nu = " + str(mnu) + " eV")
print("  Omega_k = " + str(omk))
print("  tau = " + str(tau))
print("  As = " + str(As))
print("  ns = " + str(ns))
print("Results saved to: " + csv_path)
print("First 10 rows:")
print(df.head(10).to_string(index=False))
print("Last 10 rows:")
print(df.tail(10).to_string(index=False))
