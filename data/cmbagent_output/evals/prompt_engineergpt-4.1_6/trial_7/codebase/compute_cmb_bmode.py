# filename: codebase/compute_cmb_bmode.py
import camb
import numpy as np
import pandas as pd

# Set up cosmological parameters
H0 = 67.5
ombh2 = 0.022
omch2 = 0.122
mnu = 0.06
tau = 0.06

# Initialize CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, tau=tau)
pars.InitPower.set_params(ns=0.965)
pars.set_for_lmax(3000, lens_potential_accuracy=0)

# Get results from CAMB
results = camb.get_results(pars)

# Retrieve the unlensed scalar spectra, which include TT, EE, BB, etc.
# The returned array 'cls' is of shape (number of spectra, number of l values)
# Index 2 corresponds to the B-mode spectrum (C_l^{BB})
cls = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000, spectra=['unlensed_scalar'])['unlensed_scalar']

# Create an array of multipole moments
ls = np.arange(cls.shape[1])

# We consider multipole moments from 2 to 3000
l_vals = ls[2:3001]
cl_bb = cls[2, 2:3001]

# Save results to a CSV file
data = pd.DataFrame({'l': l_vals, 'C_l_BB': cl_bb})
data.to_csv('data/result.csv', index=False)

# Print the first 10 rows for verification
print(data.head(10))
