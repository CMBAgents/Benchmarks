# filename: codebase/delensed_cmb.py
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower
import os

# Output directory
database_path = "data/"
if not os.path.exists(database_path):
    os.makedirs(database_path)

# Cosmological parameters
H0 = 67.5  # Hubble constant [km/s/Mpc]
ombh2 = 0.022  # Baryon density [dimensionless]
omch2 = 0.122  # Cold dark matter density [dimensionless]
mnu = 0.06  # Neutrino mass sum [eV]
omk = 0.0  # Curvature [dimensionless]
tau = 0.06  # Optical depth to reionization [dimensionless]
As = 2e-9  # Scalar amplitude [dimensionless]
ns = 0.965  # Scalar spectral index [dimensionless]
delensing_efficiency = 0.8  # 80% delensing

lmin = 2
lmax = 3000

# Set up CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns)
pars.set_for_lmax(lmax, lens_potential_accuracy=1)
pars.WantTensors = False

# Get results for lensed and unlensed spectra
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total', 'unlensed_scalar'])

# Extract lensed and unlensed TT spectra
lensed = powers['total'][:, 0]  # TT, lensed, in muK^2
unlensed = powers['unlensed_scalar'][:, 0]  # TT, unlensed, in muK^2

# Compute delensed TT spectrum
# Delensed: unlensed + (1 - delensing_efficiency) * (lensed - unlensed)
delensed = unlensed + (1.0 - delensing_efficiency) * (lensed - unlensed)

# Compute l(l+1)C_l/(2pi) for l=2..lmax
ells = np.arange(powers['total'].shape[0])
cl_factor = ells * (ells + 1) / (2.0 * np.pi)
TT_delensed = cl_factor * delensed

# Restrict to l=2..lmax
mask = (ells >= lmin) & (ells <= lmax)
ells_out = ells[mask]
TT_out = TT_delensed[mask]

# Save to CSV
df = pd.DataFrame({'l': ells_out, 'TT': TT_out})
csv_path = os.path.join(database_path, "result.csv")
df.to_csv(csv_path, index=False)

# Print summary
np.set_printoptions(precision=6, suppress=True)
print("Delensed CMB temperature power spectrum (l(l+1)C_l^{TT}/(2pi)) saved to " + csv_path)
print("Columns: l (multipole), TT (μK^2)")
print("First 10 rows:")
print(df.head(10).to_string(index=False))
print("\nLast 10 rows:")
print(df.tail(10).to_string(index=False))
print("\nTotal rows: " + str(len(df)))
