# filename: codebase/camb_power_spectrum.py
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
H0 = 74.0  # Hubble constant [km/s/Mpc]
ombh2 = 0.022  # Ω_b h^2
omch2 = 0.122  # Ω_c h^2
mnu = 0.06  # sum of neutrino masses [eV]
omk = 0.0  # curvature
tau = 0.06  # optical depth
As = 2e-9  # scalar amplitude
ns = 0.965  # scalar spectral index

# Set up CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns)
pars.set_for_lmax(3000, lens_potential_accuracy=0)

# Calculate results
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
totCL = powers['total']

# Extract l and TT spectrum
lmin = 2
lmax = 3000
ls = np.arange(lmin, lmax + 1)
# totCL shape: (lmax+1, 4), columns: TT, EE, BB, TE
TT = totCL[lmin:lmax+1, 0]  # TT spectrum in μK^2

# Save to CSV
df = pd.DataFrame({'l': ls, 'TT': TT})
csv_path = os.path.join(output_dir, "result.csv")
df.to_csv(csv_path, index=False)

# Print summary
np.set_printoptions(precision=6, suppress=True, linewidth=120)
print("CMB TT power spectrum (μK^2) computed for l = 2 to 3000.")
print("First 10 rows:")
print(df.head(10))
print("\nLast 10 rows:")
print(df.tail(10))
print("\nSaved to " + csv_path)