# filename: codebase/calculate_cmb_spectrum.py
import camb
from camb import model, initialpower
import numpy as np
import pandas as pd

# Set the cosmological parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=70, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965)

# Calculate the results for the given parameters
pars.set_for_lmax(3000, lens_potential_accuracy=0)
results = camb.get_results(pars)

# Get the power spectrum
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000)
totCL = powers['total']

# Extract the temperature power spectrum (TT) for l=2 to l=3000
l = np.arange(2, totCL.shape[0])
TT = totCL[2:, 0]

# Save the results to a CSV file
data = pd.DataFrame({'l': l, 'TT': TT})
data.to_csv('data/result.csv', index=False)

# Print a concise description of the results
print("CMB temperature power spectrum calculated and saved to 'data/result.csv'.")
print("The file contains multipole moments (l) from 2 to 3000 and their corresponding temperature power spectrum (TT) in microkelvin squared (\u03BCK\u00B2).")