# filename: codebase/delensed_cmb_bmode.py
import camb
from camb import model, initialpower
import numpy as np
import pandas as pd

# Set up the cosmological parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0.1)
pars.set_for_lmax(3000, lens_potential_accuracy=0)

# Calculate the results for the given parameters
results = camb.get_results(pars)

# Get the power spectra
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000)
bb_spectrum = powers['tensor']['bb']

# Apply delensing efficiency of 10%
delensing_efficiency = 0.1
delensed_bb_spectrum = bb_spectrum * (1 - delensing_efficiency)

# Prepare data for saving
l_values = np.arange(2, 3001)
bb_values = delensed_bb_spectrum[2:3001]

# Save the results to a CSV file
data = pd.DataFrame({'l': l_values, 'BB': bb_values})
data.to_csv('data/result.csv', index=False)

# Print the results to the console
print("Delensed B-mode power spectrum calculated and saved to 'data/result.csv'.")
print(data.head())