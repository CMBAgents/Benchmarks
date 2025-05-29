# filename: codebase/E_mode_power_spectrum.py
import camb
from camb import model, initialpower
import numpy as np
import pandas as pd

# Set the cosmological parameters
H0 = 67.5  # Hubble constant in km/s/Mpc
ombh2 = 0.022  # Baryon density
omch2 = 0.122  # Cold dark matter density
tau = 0.1  # Optical depth to reionization
As = 1.8e-9 * np.exp(2 * tau)  # Scalar amplitude
ns = 0.95  # Scalar spectral index

# Initialize CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
pars.InitPower.set_params(As=As, ns=ns)
pars.set_for_lmax(100, lens_potential_accuracy=0)

# Calculate the results
results = camb.get_results(pars)

# Get the power spectrum
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=100)
EE_spectrum = powers['EE']

# Extract the multipole moments and E-mode power spectrum
l_values = np.arange(2, 101)
EE_values = EE_spectrum[2:101, 0] * l_values * (l_values + 1) / (2 * np.pi)

# Save the results to a CSV file
df = pd.DataFrame({'l': l_values, 'EE': EE_values})
df.to_csv('data/result.csv', index=False)

# Print the results
print("E-mode polarization power spectrum calculated and saved to 'data/result.csv'.")
print(df)
