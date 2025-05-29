# filename: codebase/compute_cmb_EE_spectrum.py
import os
import numpy as np
import camb
import csv

# Set up the CAMB parameters for a flat LambdaCDM cosmology
pars = camb.CAMBparams()
# Set cosmological parameters
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)
# Set the maximum multipole order, here using 150 for safety
pars.set_for_lmax(150, lens_potential_accuracy=0)

# Run CAMB and get results
results = camb.get_results(pars)
# Get the total CMB power spectra; the unit is set to microK
# The returned array has columns: TT, EE, BB, TE
powers = results.get_total_cls(lmax=150, CMB_unit='muK')

# Create array of multipoles
ls = np.arange(powers.shape[0])

# Prepare list to hold output data for l=2 to l=100
output_data = []
for l in range(2, 101):
    # Compute scaled EE power spectrum: l*(l+1)C_l^{EE}/(2*pi)
    EE_value = l * (l + 1) / (2 * np.pi) * powers[l, 1]
    output_data.append([l, EE_value])

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

# Write the output data to a CSV file
with open(os.path.join('data', 'result.csv'), mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['l', 'EE'])
    for row in output_data:
        writer.writerow(row)

# Print detailed results to the console
for row in output_data:
    print('l = ' + str(row[0]) + ', EE = ' + str(row[1]))