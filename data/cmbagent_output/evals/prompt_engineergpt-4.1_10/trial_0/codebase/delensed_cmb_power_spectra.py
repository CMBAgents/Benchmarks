# filename: codebase/delensed_cmb_power_spectra.py
r"""
Compute the delensed CMB temperature power spectrum for a flat Lambda CDM cosmology using CAMB.

Parameters:
    H0: Hubble constant [km/s/Mpc]
    ombh2: Baryon density parameter [dimensionless]
    omch2: Cold dark matter density parameter [dimensionless]
    mnu: Sum of neutrino masses [eV]
    omk: Curvature parameter [dimensionless]
    tau: Optical depth to reionization [dimensionless]
    As: Scalar amplitude [dimensionless]
    ns: Scalar spectral index [dimensionless]
    delensing_efficiency: Fraction of lensing removed [dimensionless, 0-1]

Output:
    CSV file with columns:
        l: Multipole moment (integer, 2 to 3000)
        TT: Delensed temperature power spectrum [l(l+1)C_l^{TT}/(2pi) in microK^2]
"""

import numpy as np
import pandas as pd
import camb
from camb import model, initialpower
import os

# Output directory and file
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_file = os.path.join(output_dir, "result.csv")

# Cosmological parameters
H0 = 67.5  # Hubble constant [km/s/Mpc]
ombh2 = 0.022  # Baryon density [dimensionless]
omch2 = 0.122  # Cold dark matter density [dimensionless]
mnu = 0.06  # Neutrino mass sum [eV]
omk = 0.0  # Curvature [dimensionless]
tau = 0.06  # Optical depth [dimensionless]
As = 2e-9  # Scalar amplitude [dimensionless]
ns = 0.965  # Scalar spectral index [dimensionless]
delensing_efficiency = 0.8  # 80% delensing

# Set up CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns)
pars.set_for_lmax(3000, lens_potential_accuracy=1)
pars.WantTensors = False
pars.Want_CMB_lensing = True

# Calculate results for these parameters
results = camb.get_results(pars)

# Get lensed and unlensed CMB power spectra
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000, raw_cl=False)
lensed_cl = powers['total']  # lensed TT, EE, BB, TE
unlensed_cl = powers['unlensed_scalar']  # unlensed TT, EE, BB, TE

# Multipole range
lmin = 2
lmax = 3000
ells = np.arange(lmin, lmax + 1)

# Extract TT spectra (column 0), shape: (lmax+1, 4)
# lensed_cl and unlensed_cl are indexed by l, starting from l=0
lensed_TT = lensed_cl[lmin:lmax+1, 0]  # [muK^2]
unlensed_TT = unlensed_cl[lmin:lmax+1, 0]  # [muK^2]

# Delensing: remove 80% of the lensing contribution
# delensed_TT = unlensed_TT + (1 - delensing_efficiency) * (lensed_TT - unlensed_TT)
delensed_TT = unlensed_TT + (1.0 - delensing_efficiency) * (lensed_TT - unlensed_TT)

# Compute l(l+1)C_l/(2pi) [muK^2]
factor = ells * (ells + 1) / (2.0 * np.pi)
delensed_TT_Cl = factor * delensed_TT  # [muK^2]

# Save to CSV
df = pd.DataFrame({'l': ells, 'TT': delensed_TT_Cl})
df.to_csv(output_file, index=False)

# Print summary
np.set_printoptions(precision=6, suppress=True)
print("Delensed CMB temperature power spectrum (l(l+1)C_l^{TT}/(2pi)) saved to data/result.csv")
print("First 10 rows:")
print(df.head(10).to_string(index=False))
print("\nLast 10 rows:")
print(df.tail(10).to_string(index=False))
print("\nColumns:")
print(list(df.columns))
print("Units: l (dimensionless), TT (microK^2)")
