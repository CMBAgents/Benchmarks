# filename: codebase/delensed_BB.py
import os
import camb
import numpy as np
import csv


def main():
    # Set up CAMB parameters for a flat Lambda CDM cosmology
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(ns=0.965)
    
    # Set the maximum multipole
    pars.set_for_lmax(2000, lens_potential_accuracy=1)
    
    # Get results from CAMB
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL = powers['total']
    
    # totCL contains the power spectra; the columns are:
    # 0: TT, 1: EE, 2: TE, 3: BB
    # Create an array for the multipole moments. CAMB returns data starting at l=2
    ls = np.arange(totCL.shape[0])
    BB = totCL[:, 3]
    
    # Apply 10% delensing efficiency
    delensing_eff = 0.1
    BB_delensed = BB * (1 - delensing_eff)
    
    # Ensure the output directory exists
    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the results to CSV
    output_file = os.path.join(output_dir, 'result.csv')
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['l', 'BB'])
        for l, bb in zip(ls, BB_delensed):
            writer.writerow([str(l), str(bb)])
    
    # Print the first and last 10 rows and the total number of multipoles
    combined = list(zip(ls, BB_delensed))
    print('First 10 rows:')
    for row in combined[:10]:
        print(row)
    
    print('Last 10 rows:')
    for row in combined[-10:]:
        print(row)
    
    print('Total number of multipoles: ' + str(len(ls)))


if __name__ == '__main__':
    main()