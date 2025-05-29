# filename: codebase/compute_cmb_power_spectrum.py
import os
import camb
from camb import model, initialpower
import numpy as np
import pandas as pd

# Set output directory
output_dir = os.path.join('data')
os.makedirs(output_dir, exist_ok=True)

# Set up CAMB parameters for a flat Lambda CDM cosmology
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)
pars.set_for_lmax(2500, lens_potential_accuracy=0)

# Get results from CAMB and compute the CMB power spectra
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, lmax=2500, CMB_unit='muK')
cmb_result = powers['total']

# Generate multipole moments and extract C_l^{TT} from the result
l = np.arange(cmb_result.shape[0])
cl_tt = cmb_result[:, 0]

# Create a DataFrame and save the results to a CSV file
df = pd.DataFrame({'l': l, 'C_l^{TT}': cl_tt})
csv_path = os.path.join(output_dir, 'result.csv')
df.to_csv(csv_path, index=False)

# Print the first and last five rows of the DataFrame for verification
print('First five rows:')
print(df.head())
print('\nLast five rows:')
print(df.tail())