# filename: codebase/compute_cmb_power_spectrum.py
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
ombh2 = 0.02  # Omega_b h^2 [dimensionless]
omch2 = 0.122  # Omega_c h^2 [dimensionless]
mnu = 0.06  # Sum of neutrino masses [eV]
omk = 0.0  # Curvature [dimensionless]
tau = 0.06  # Optical depth to reionization [dimensionless]
As = 2e-9  # Scalar amplitude [dimensionless]
ns = 0.965  # Scalar spectral index [dimensionless]

# Set up CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns)
pars.set_for_lmax(3000, lens_potential_accuracy=0)

# Calculate results
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000)
totCL = powers['total']

# l values: CAMB returns from l=0, so index 2 corresponds to l=2
lmin = 2
lmax = 3000
ells = np.arange(lmin, lmax + 1)  # l=2 to l=3000

# Compute l(l+1)C_l^{TT}/(2π) in μK^2
cl_tt = totCL[lmin:lmax+1, 0]  # TT spectrum, μK^2
llp1 = ells * (ells + 1)
cl_tt_scaled = llp1 * cl_tt / (2.0 * np.pi)  # μK^2

# Save to CSV
df = pd.DataFrame({'l': ells, 'TT': cl_tt_scaled})
csv_path = os.path.join(output_dir, "result.csv")
df.to_csv(csv_path, index=False)

# Print summary of results
pd.set_option("display.precision", 6)
pd.set_option("display.width", 120)
print("CMB temperature power spectrum (l(l+1)C_l^{TT}/(2π)) in μK^2, l=2 to l=3000")
print("First 5 rows:\n" + str(df.head()))
print("Last 5 rows:\n" + str(df.tail()))
print("Saved results to " + csv_path)