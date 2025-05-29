# filename: codebase/cmb_bmode_power_spectrum.py
import camb
from camb import model, initialpower
import numpy as np
import pandas as pd

# Set the cosmological parameters
H0 = 67.5  # Hubble constant in km/s/Mpc
ombh2 = 0.022  # Baryon density
omch2 = 0.122  # Cold dark matter density
mnu = 0.06  # Neutrino mass sum in eV
omk = 0.0  # Curvature
tau = 0.06  # Optical depth to reionization
tensor_to_scalar_ratio = 0.1  # Tensor-to-scalar ratio
As = 2e-9  # Scalar amplitude
ns = 0.965  # Scalar spectral index

# Initialize CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns, r=tensor_to_scalar_ratio)
pars.set_for_lmax(3000, lens_potential_accuracy=0)

# Calculate the results
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000)

# Extract the B-mode power spectrum
bb_spectrum = powers['tensor'][:, 2]  # B-mode is the third column in the tensor spectrum

# Prepare data for saving
l_values = np.arange(2, len(bb_spectrum))
bb_values = bb_spectrum[2:]

# Save to CSV
df = pd.DataFrame({'l': l_values, 'BB': bb_values})
df.to_csv('data/result.csv', index=False)

# Print the results
print("B-mode polarization power spectrum (C_l^{BB}) calculated and saved to 'data/result.csv'.")
print("The file contains multipole moments (l) from 2 to 3000 and their corresponding B-mode power spectrum values in microkelvin squared (μK^2).")