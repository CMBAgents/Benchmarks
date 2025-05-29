# filename: codebase/camb_cmb_bmode.py
import camb
from camb import model, initialpower
import numpy as np
import pandas as pd

# Set up the cosmological parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(3000, lens_potential_accuracy=0)

# Calculate the results
results = camb.get_results(pars)

# Get the power spectrum
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000)
bb_spectrum = powers['tensor'][:, 2]  # B-mode polarization

# Calculate l(l+1)C_l^{BB}/(2*pi)
l = np.arange(2, bb_spectrum.size)
bb_power_spectrum = l * (l + 1) * bb_spectrum[2:] / (2 * np.pi)

# Save the results to a CSV file
df = pd.DataFrame({'l': l, 'BB': bb_power_spectrum})
df.to_csv('data/result.csv', index=False)

# Print the results
print("CMB B-mode polarization power spectrum calculated and saved to 'data/result.csv'.")
print("First few rows of the result:")
print(df.head())