# filename: codebase/cmb_power_spectrum.py
import camb
from camb import model, initialpower
import numpy as np
import pandas as pd

# Set the cosmological parameters
H0 = 74.0  # Hubble constant in km/s/Mpc
ombh2 = 0.022  # Baryon density
omch2 = 0.122  # Cold dark matter density
mnu = 0.06  # Neutrino mass sum in eV
omk = 0.0  # Curvature
tau = 0.06  # Optical depth to reionization
As = 2e-9  # Scalar amplitude
ns = 0.965  # Scalar spectral index

# Initialize CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns)
pars.set_for_lmax(3000, lens_potential_accuracy=0)

# Calculate the results
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000)
totCL = powers['total']

# Extract the temperature power spectrum
l = np.arange(2, totCL.shape[0])
TT = totCL[2:, 0]

# Save the results to a CSV file
data = pd.DataFrame({'l': l, 'TT': TT})
data.to_csv('data/result.csv', index=False)

# Print the results
print("CMB temperature power spectrum calculated and saved to 'data/result.csv'.")
print("First few rows of the data:")
print(data.head())