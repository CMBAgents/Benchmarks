# filename: codebase/camb_cmb_spectrum.py
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
ombh2 = 0.02  # Omega_b h^2 [unitless]
omch2 = 0.122  # Omega_c h^2 [unitless]
mnu = 0.06  # Sum of neutrino masses [eV]
omk = 0.0  # Curvature [unitless]
tau = 0.06  # Optical depth [unitless]
As = 2e-9  # Scalar amplitude [unitless]
ns = 0.965  # Scalar spectral index [unitless]

# Set up CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns)
pars.set_for_lmax(3000, lens_potential_accuracy=0)

# Calculate results
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')

# Extract the total (lensed) CMB power spectrum
totCL = powers['total']  # shape: (lmax+1, 4), columns: TT, EE, BB, TE

# Multipole moments
lmax = 3000
ells = np.arange(totCL.shape[0])  # 0 to lmax

# Compute l(l+1)C_l^{TT}/(2pi) in microkelvin^2
TT = totCL[:, 0]  # TT spectrum [muK^2]
cl_factor = ells * (ells + 1) / (2.0 * np.pi)
TT_spectrum = cl_factor * TT  # [muK^2]

# Select l=2 to l=3000
lmin = 2
lmax = 3000
mask = (ells >= lmin) & (ells <= lmax)
ells_out = ells[mask]
TT_out = TT_spectrum[mask]

# Save to CSV
df = pd.DataFrame({'l': ells_out, 'TT': TT_out})
csv_path = os.path.join(output_dir, "result.csv")
df.to_csv(csv_path, index=False)

# Print summary
np.set_printoptions(precision=6, suppress=True, linewidth=120)
print("CMB temperature power spectrum (l(l+1)C_l^{TT}/(2pi)) computed for l=2 to l=3000.")
print("Results saved to: " + csv_path)
print("First 10 rows:")
print(df.head(10).to_string(index=False))
print("\nLast 10 rows:")
print(df.tail(10).to_string(index=False))
print("\nColumn units:")
print("l: Multipole moment (dimensionless)")
print("TT: Temperature power spectrum [microkelvin^2] (l(l+1)C_l^{TT}/(2pi))")
