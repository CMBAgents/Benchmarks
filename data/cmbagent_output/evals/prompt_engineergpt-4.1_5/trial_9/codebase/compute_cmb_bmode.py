# filename: codebase/compute_cmb_bmode.py
r"""
Compute the CMB B-mode polarization power spectrum (l(l+1)C_l^{BB}/(2pi)) for a flat Lambda CDM cosmology
using CAMB, with the following parameters:
- Hubble constant (H0): 67.5 km/s/Mpc
- Baryon density (ombh2): 0.022
- Cold dark matter density (omch2): 0.122
- Neutrino mass sum (mnu): 0.06 eV
- Curvature (omk): 0
- Optical depth to reionization (tau): 0.06
- Tensor-to-scalar ratio (r): 0
- Scalar amplitude (As): 2e-9
- Scalar spectral index (ns): 0.965

The B-mode power spectrum is computed for multipole moments l=2 to l=3000, in units of microkelvin^2.
Results are saved in 'data/result.csv' with columns: l, BB.

Requires: camb, numpy, pandas
"""

import os
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

# Ensure data directory exists
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Cosmological parameters
H0 = 67.5  # Hubble constant [km/s/Mpc]
ombh2 = 0.022  # Baryon density parameter [dimensionless]
omch2 = 0.122  # Cold dark matter density parameter [dimensionless]
mnu = 0.06  # Neutrino mass sum [eV]
omk = 0.0  # Curvature [dimensionless]
tau = 0.06  # Optical depth to reionization [dimensionless]
r = 0.0  # Tensor-to-scalar ratio [dimensionless]
As = 2e-9  # Scalar amplitude [dimensionless]
ns = 0.965  # Scalar spectral index [dimensionless]

# Multipole range
lmin = 2
lmax = 3000

# Set up CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns, r=r)
pars.set_for_lmax(lmax, lens_potential_accuracy=1)
pars.WantTensors = True  # Needed for BB spectrum, even if r=0

# Calculate results
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total'])

# Extract BB spectrum
cl = powers['total']  # shape: (lmax+1, 4) for TT, EE, BB, TE
# cl[:,2] is BB
ells = np.arange(cl.shape[0])  # l = 0, 1, ..., lmax
BB = cl[:,2]  # BB spectrum in muK^2

# Compute l(l+1)C_l^{BB}/(2pi)
factor = ells * (ells + 1) / (2.0 * np.pi)
BB_power = factor * BB  # units: muK^2

# Select l=2 to l=3000
mask = (ells >= lmin) & (ells <= lmax)
ells_out = ells[mask]
BB_out = BB_power[mask]

# Save to CSV
df = pd.DataFrame({'l': ells_out, 'BB': BB_out})
csv_path = os.path.join(output_dir, "result.csv")
df.to_csv(csv_path, index=False)

# Print summary
pd.set_option("display.precision", 8)
print("CMB B-mode polarization power spectrum (l(l+1)C_l^{BB}/(2pi)) computed for l=2 to l=3000.")
print("Units: BB in microkelvin^2 (muK^2).")
print("First 10 rows of the result:")
print(df.head(10))
print("\nSaved full results to " + csv_path)