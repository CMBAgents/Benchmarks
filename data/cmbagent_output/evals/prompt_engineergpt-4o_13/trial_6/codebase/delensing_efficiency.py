# filename: codebase/delensing_efficiency.py
import numpy as np
import pandas as pd
from camb import CAMBparams, get_results

# Load lensing noise power spectrum
n0_data = pd.read_csv('/Users/antoidicherianlonappan/Software/Benchmarks/data/N0.csv')
l = n0_data['l'].values
nl = n0_data['Nl'].values
n0 = nl * (2 * np.pi) / (l * (l + 1))**2

# Set up CAMB parameters
params = CAMBparams()
params.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
params.InitPower.set_params(As=2e-9, ns=0.965)
params.set_for_lmax(2000, lens_potential_accuracy=1)

# Get results from CAMB
results = get_results(params)
cl = results.get_cmb_power_spectra(params, lmax=2000, CMB_unit='muK')
cl_bb = cl['total'][:, 2]  # Lensed B-mode power spectrum
cl_pp = results.get_lens_potential_cls(lmax=2000)[:, 0] * (2 * np.pi) / (l * (l + 1))**2

# Ensure cl_pp and n0 have the same length
cl_pp = cl_pp[:len(n0)]

# Compute residual lensing potential power spectrum
cl_pp_res = cl_pp * (1 - (cl_pp / (cl_pp + n0)))

# Compute delensed B-mode power spectrum
cl_bb_delensed = cl_bb * (1 - cl_pp_res / cl_pp)

# Compute delensing efficiency
delensing_efficiency = 100 * (cl_bb - cl_bb_delensed) / cl_bb

# Save results for l=2 to l=100
result_df = pd.DataFrame({
    'l': np.arange(2, 101),
    'delensing_efficiency': delensing_efficiency[2:101]
})
result_df.to_csv('data/result.csv', index=False)

print("Delensing efficiency calculated and saved for multipole moments l=2 to l=100.")
