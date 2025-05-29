# filename: codebase/cmb_power_spectrum.py
import camb
from camb import model, initialpower
import numpy as np
import pandas as pd

# Set the cosmological parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.3, ombh2=0.022, omch2=0.122, omk=0.05, mnu=0.06, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965)

# Set the range of multipole moments
pars.set_for_lmax(3000, lens_potential_accuracy=0)

# Calculate the results
results = camb.get_results(pars)

# Get the CMB power spectra
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000)
totCL = powers['total']

# Calculate the temperature power spectrum
ell = np.arange(2, totCL.shape[0])
TT = totCL[2:, 0] * (ell * (ell + 1)) / (2 * np.pi)

# Save the results to a CSV file
data = pd.DataFrame({'l': ell, 'TT': TT})
data.to_csv('data/result.csv', index=False)

# Print a concise description of the results
print("CMB temperature power spectrum calculated and saved to 'data/result.csv'.")
print("The file contains multipole moments (l) from 2 to 3000 and the corresponding temperature power spectrum (TT) in units of μK^2.")