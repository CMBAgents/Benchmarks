# filename: codebase/camb_bb_spectrum.py
r"""
Compute the CMB B-mode polarization power spectrum (C_l^{BB}) for a flat Lambda CDM cosmology
using CAMB, with the following parameters:
- Hubble constant (H0): 67.5 km/s/Mpc
- Baryon density (ombh2): 0.022
- Cold dark matter density (omch2): 0.122
- Neutrino mass sum (mnu): 0.06 eV
- Curvature (omk): 0
- Optical depth to reionization (tau): 0.06
- Tensor-to-scalar ratio (r): 0.1
- Scalar amplitude (As): 2e-9
- Scalar spectral index (ns): 0.965

The B-mode power spectrum (C_l^{BB}) is computed for multipole moments l=2 to l=3000,
in units of microkelvin squared (\u03BCK^2), and saved to 'data/result.csv'.

Requires: camb, numpy, pandas
"""

import os
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

# Ensure output directory exists
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
r = 0.1  # Tensor-to-scalar ratio [dimensionless]
As = 2e-9  # Scalar amplitude [dimensionless]
ns = 0.965  # Scalar spectral index [dimensionless]

# Multipole range
lmin = 2
lmax = 3000

# Set up CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns, r=r)
pars.WantTensors = True
pars.set_for_lmax(lmax, lens_potential_accuracy=0)

# Get results and power spectra
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
# 'raw_cl=True' gives Cl, not Dl, in units of muK^2

# Extract the total power spectrum
totCL = powers['total']  # shape: (lmax+1, 4) for TT, EE, BB, TE

# l values: CAMB returns Cls for l=0 to lmax
ells = np.arange(totCL.shape[0])  # l=0 to lmax

# Extract BB spectrum for l=2 to l=3000
l_mask = (ells >= lmin) & (ells <= lmax)
ells_out = ells[l_mask]
BB = totCL[l_mask, 2]  # BB is the third column (index 2), units: muK^2

# Save to CSV
df = pd.DataFrame({'l': ells_out, 'BB': BB})
csv_path = os.path.join(output_dir, "result.csv")
df.to_csv(csv_path, index=False)

# Print summary of results
pd.set_option("display.precision", 6)
pd.set_option("display.max_rows", 10)
print("CMB B-mode polarization power spectrum (C_l^{BB}) computed for l=2 to l=3000.")
print("Units: l (dimensionless), BB (muK^2)")
print("First and last 5 rows of the result:")
print(pd.concat([df.head(5), df.tail(5)]))
print("\nResults saved to " + csv_path)