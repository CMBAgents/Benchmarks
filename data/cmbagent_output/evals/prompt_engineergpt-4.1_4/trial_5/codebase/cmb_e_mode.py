# filename: codebase/cmb_e_mode.py
import camb
import numpy as np
import pandas as pd

# Define parameters for multipole range
lmin = 2
lmax = 3000

# Setup CAMB parameters for a flat Lambda CDM cosmology
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)
pars.set_for_lmax(lmax, lens_potential_accuracy=0)

# Run CAMB and get results
results = camb.get_results(pars)
# Get total power spectra; column 1 corresponds to the E-mode polarization
powers = results.get_total_cls(lmax)

# Create an array for multipole moments from lmin to lmax
ls = np.arange(lmin, lmax + 1)
EE = powers[lmin:lmax + 1, 1]

# Create a DataFrame with the results
df = pd.DataFrame({'l': ls, 'EE': EE})

# Save the results to CSV in the data directory
df.to_csv('data/result.csv', index=False)

# Print verification details: first 5 rows, last 5 rows, and total number of multipoles
print('First 5 rows:')
print(df.head())
print('Last 5 rows:')
print(df.tail())
print('Total multipoles:', len(df))