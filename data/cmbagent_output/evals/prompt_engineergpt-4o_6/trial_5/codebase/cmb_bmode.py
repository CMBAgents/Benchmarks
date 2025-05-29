# filename: codebase/cmb_bmode.py
import camb
from camb import model, initialpower
import numpy as np
import pandas as pd

# Set up the cosmological parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0.1)
pars.set_for_lmax(3000, lens_potential_accuracy=0)

# Calculate the results
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000)

# Extract the B-mode power spectrum
bb_spectrum = powers['tensor'][:, 2]  # B-mode is the third column in the tensor power spectrum

# Prepare data for saving
l_values = np.arange(2, 3001)
bb_values = bb_spectrum[2:3001]

# Save to CSV
df = pd.DataFrame({'l': l_values, 'BB': bb_values})
df.to_csv('data/result.csv', index=False)

# Print the results
print("B-mode polarization power spectrum calculated and saved to 'data/result.csv'.")
print("First few entries:")
print(df.head())