# filename: codebase/cmb_power_spectrum.py
import numpy as np
import camb
from camb import model, initialpower
import pandas as pd

# Set up parameters for flat LambdaCDM
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965)
pars.set_for_lmax(3000, lens_potential_accuracy=0)

# Obtain results and compute the total CMB power spectrum
results = camb.get_results(pars)
# totCL contains TT, EE, BB, TE spectra with shape (lmax+1, 4)
totCL = results.get_total_cls(lmax=3000, CMB_unit='muK')

# Define the multipole moments from 2 to 3000
l_vals = np.arange(2, 3001)
# Extract the TT spectrum for l=2 to l=3000 (TT is in the first column)
TT = totCL[2:3001, 0]

# Save the results to a CSV file
df = pd.DataFrame({'l': l_vals, 'TT': TT})
df.to_csv('data/result.csv', index=False)

# Print the first and last five rows for verification
print('First five rows:')
print(df.head())
print('Last five rows:')
print(df.tail())