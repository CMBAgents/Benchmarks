# filename: codebase/camb_cmb_E_mode.py
import camb
from camb import model
import numpy as np
import pandas as pd

# Set up CAMB parameters for flat Lambda CDM model
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(As=2e-9, ns=0.965)
pars.set_for_lmax(3000, lens_potential_accuracy=0)

# Get results from CAMB and compute CMB power spectra
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, lmax=3000, CMB_unit='muK', spectra=['EE'])

# Extract the E-mode polarization power spectrum
# powers['EE'] is an array where the index corresponds to the multipole moment l
# We take l from 2 to 3000
l_vals = np.arange(2, 3001)
ee_vals = powers['EE'][2:3001, 0]

# Create a DataFrame and save the results
df = pd.DataFrame({'l': l_vals, 'EE': ee_vals})
df.to_csv('data/result.csv', index=False)

# Print first 10 rows, last 10 rows, and the total number of rows for verification
print('First 10 rows:')
print(df.head(10))

print('\nLast 10 rows:')
print(df.tail(10))

print('\nTotal number of rows: ' + str(len(df)))
