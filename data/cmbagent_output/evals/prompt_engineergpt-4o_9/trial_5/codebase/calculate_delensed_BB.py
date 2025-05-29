# filename: codebase/calculate_delensed_BB.py
import camb
from camb import model, initialpower
import numpy as np
import pandas as pd

# Set the cosmological parameters
H0 = 67.5  # Hubble constant in km/s/Mpc
ombh2 = 0.022  # Baryon density
omch2 = 0.122  # Cold dark matter density
mnu = 0.06  # Neutrino mass sum in eV
omk = 0  # Curvature
tau = 0.06  # Optical depth to reionization
r = 0.1  # Tensor-to-scalar ratio
As = 2e-9  # Scalar amplitude
ns = 0.965  # Scalar spectral index

# Initialize CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns, r=r)
pars.set_for_lmax(3000, lens_potential_accuracy=0)

# Calculate the results
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000)

# Extract the B-mode power spectrum
bb_spectrum = powers['tensor'][:, 2]  # B-mode is the third column in the tensor spectrum

# Apply delensing efficiency
delensing_efficiency = 0.1
delensed_bb_spectrum = bb_spectrum * (1 - delensing_efficiency)

# Prepare data for saving
l_values = np.arange(2, 3001)
data = {'l': l_values, 'BB': delensed_bb_spectrum[2:3001]}

# Save to CSV
df = pd.DataFrame(data)
df.to_csv('data/result.csv', index=False)

# Print the results
print("Delensed B-mode power spectrum calculated and saved to 'data/result.csv'.")
print("First few rows of the data:")
print(df.head())