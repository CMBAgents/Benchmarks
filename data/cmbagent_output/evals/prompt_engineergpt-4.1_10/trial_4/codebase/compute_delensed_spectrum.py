# filename: codebase/compute_delensed_spectrum.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

# Set up cosmological parameters for a flat Lambda CDM cosmology
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, tau=0.06)
pars.InitPower.set_params(ns=0.965, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=1)

# Compute results for both lensed and unlensed spectra
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=2500, raw_cl=True)

# Extract unlensed and lensed TT power spectra
# unlensed_cl and lensed_cl correspond to the TT spectrum from unlensed_scalar and total respectively
unlensed_cl = powers['unlensed_scalar'][:, 0]  # TT spectrum from unlensed scalar perturbations
lensed_cl = powers['total'][:, 0]               # TT spectrum including lensing

# Create multipole array l
ells = np.arange(powers['total'].shape[0])

# Apply 80% delensing efficiency:
# delensed_TT = TT_unlensed + 0.2*(TT_lensed - TT_unlensed)
# This means that 80% of the lensing effect is removed
delensed_tt = unlensed_cl + 0.2 * (lensed_cl - unlensed_cl)

# Save the delensed power spectrum to CSV in the data folder
data_df = pd.DataFrame({'l': ells, 'TT': delensed_tt})
os.makedirs('data', exist_ok=True)
data_df.to_csv('data/result.csv', index=False)

# Print the first and last 10 rows for verification
print("First 10 rows of the delensed spectrum:")
print(data_df.head(10))

print("\nLast 10 rows of the delensed spectrum:")
print(data_df.tail(10))
