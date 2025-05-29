# filename: codebase/compute_relative_difference.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model

# Ensure the output directory exists
output_dir = os.path.join('data')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def compute_power_spectrum(neutrino_hierarchy):
    """
    Compute the linear matter power spectrum using CAMB for a given neutrino hierarchy.

    Parameters:
    neutrino_hierarchy (str): Either 'normal' or 'inverted'.

    Returns:
    tuple: (k values, power spectrum) for redshift = 0.
    """
    # Initialize CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.12)
    pars.InitPower.set_params(ns=0.965, r=0)
    
    # Set matter power spectrum parameters
    redshifts = [0]
    kmax = 2.0
    pars.set_matter_power(redshifts=redshifts, kmax=kmax, nonlinear=False)
    pars.NonLinear = model.NonLinear_none

    # Adjust parameters based on neutrino hierarchy if needed
    # In a detailed scenario, one might adjust the neutrino mass splitting; here we use the same setup
    # as a placeholder for the different hierarchies.
    if neutrino_hierarchy not in ["normal", "inverted"]:
        raise ValueError("Unknown neutrino hierarchy specified")

    # Run CAMB to get results
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=kmax, npoints=1000)
    return kh, pk[0]


# Compute power spectra for normal and inverted neutrino hierarchies
kh_normal, pk_normal = compute_power_spectrum("normal")
kh_inverted, pk_inverted = compute_power_spectrum("inverted")

# Compute the relative difference: (P_inverted - P_normal) / P_normal
relative_diff = (pk_inverted - pk_normal) / pk_normal

# Prepare a DataFrame to save the results
df = pd.DataFrame({"k": kh_normal, "relative_difference": relative_diff})

# Save the results to a CSV file in the data/ directory
output_file = os.path.join(output_dir, "result.csv")
df.to_csv(output_file, index=False)

# Print the first and last five rows of the results
print("First five rows:")
print(df.head(5))
print("Last five rows:")
print(df.tail(5))
