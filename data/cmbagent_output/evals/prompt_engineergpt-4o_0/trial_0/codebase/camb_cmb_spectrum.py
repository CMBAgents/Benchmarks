# filename: codebase/camb_cmb_spectrum.py
import camb
from camb import model, initialpower
import numpy as np
import pandas as pd

# Set the cosmological parameters
H0 = 67.5  # Hubble constant in km/s/Mpc
ombh2 = 0.02  # Baryon density
omch2 = 0.122  # Cold dark matter density
mnu = 0.06  # Neutrino mass sum in eV
omk = 0.0  # Curvature (flat cosmology)
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

# Extract the temperature power spectrum
totCL = powers['total']
ls = np.arange(2, totCL.shape[0])
cl_tt = totCL[2:, 0]

# Calculate the temperature power spectrum in units of muK^2
cl_tt_muK2 = (ls * (ls + 1) * cl_tt) / (2 * np.pi)

# Save the results to a CSV file
df = pd.DataFrame({'l': ls, 'TT': cl_tt_muK2})
df.to_csv('data/result.csv', index=False)

print("CMB temperature power spectrum calculated and saved to 'data/result.csv'.")