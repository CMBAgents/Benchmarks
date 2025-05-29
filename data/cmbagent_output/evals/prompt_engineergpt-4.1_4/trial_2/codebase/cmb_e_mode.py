# filename: codebase/cmb_e_mode.py
import numpy as np
import camb
from camb import model, initialpower
import pandas as pd


def compute_cmb_e_mode():
    # Set up parameters for a flat Lambda CDM cosmology
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(ns=0.965, r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0)

    # Compute results for the CMB power spectra
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL = powers['total']

    # Extract multipole moments and E-mode polarization (EE) from totCL
    ells = np.arange(totCL.shape[0])
    ee = totCL[:, 1]  # EE column

    # Create a DataFrame with the required columns
    df = pd.DataFrame({'Multipole': ells, 'E-mode Polarization': ee})

    # Save the DataFrame to a CSV file in the data directory
    df.to_csv('../data/result.csv', index=False)

    # Print first and last few rows for verification
    print('First few rows:')
    print(df.head())
    print('Last few rows:')
    print(df.tail())


if __name__ == '__main__':
    compute_cmb_e_mode()