# filename: codebase/calculate_EE_powerspectrum.py
import camb
from camb import model, initialpower
import numpy as np
import pandas as pd

# Set up the cosmological parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.04)
pars.InitPower.set_params(As=2e-9, ns=0.965)

# Calculate the power spectra
pars.set_for_lmax(3000, lens_potential_accuracy=0)
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000)

# Extract the E-mode polarization power spectrum
ee_spectrum = powers['EE']
l_values = np.arange(2, ee_spectrum.shape[0])

# Calculate l(l+1)C_l^{EE}/(2*pi)
ee_power_spectrum = l_values * (l_values + 1) * ee_spectrum[2:, 0] / (2 * np.pi)

# Save the results to a CSV file
df = pd.DataFrame({'l': l_values, 'EE': ee_power_spectrum})
df.to_csv('data/result.csv', index=False)

# Print a concise description of the results
print("E-mode polarization power spectrum calculated and saved to 'data/result.csv'.")
print("The file contains multipole moments (l) from 2 to 3000 and their corresponding E-mode power spectrum (EE) in units of μK^2.")
