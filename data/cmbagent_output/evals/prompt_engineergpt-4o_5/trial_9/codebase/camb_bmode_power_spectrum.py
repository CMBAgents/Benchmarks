# filename: codebase/camb_bmode_power_spectrum.py
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
As = 2e-9  # Scalar amplitude
ns = 0.965  # Scalar spectral index
r = 0  # Tensor-to-scalar ratio

# Initialize CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns, r=r)
pars.set_for_lmax(3000, lens_potential_accuracy=0)

# Calculate the results
results = camb.get_results(pars)

# Get the power spectrum
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000)
bb_spectrum = powers['tensor'][:, 2]  # B-mode polarization power spectrum

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