# filename: codebase/camb_neutrino_power_spectrum.py
import camb
from camb import model, initialpower
import numpy as np
import pandas as pd

# Set cosmological parameters
H0 = 67.5  # Hubble constant in km/s/Mpc
ombh2 = 0.022  # Baryon density
omch2 = 0.122  # Cold dark matter density
As = 2e-9  # Scalar amplitude
ns = 0.965  # Scalar spectral index
sum_nu_mass = 0.11  # Neutrino mass sum in eV

# Function to calculate power spectrum for a given neutrino hierarchy
def calculate_power_spectrum(hierarchy):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=sum_nu_mass, omk=0, tau=0.06)
    pars.InitPower.set_params(As=As, ns=ns, r=0)
    pars.set_matter_power(redshifts=[0], kmax=2.0)
    
    if hierarchy == 'normal':
        pars.set_neutrino_hierarchy('normal')
    elif hierarchy == 'inverted':
        pars.set_neutrino_hierarchy('inverted')
    
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=2.0, npoints=200)
    return kh, pk[0]

# Calculate power spectra for both hierarchies
k_values, pk_normal = calculate_power_spectrum('normal')
_, pk_inverted = calculate_power_spectrum('inverted')

# Calculate relative difference
rel_diff = (pk_inverted / pk_normal) - 1

# Save results to CSV
results_df = pd.DataFrame({'k': k_values, 'rel_diff': rel_diff})
results_df.to_csv('data/result.csv', index=False)

# Print results
print("Relative difference in the linear matter power spectrum has been calculated and saved to 'data/result.csv'.")
print("First few rows of the results:")
print(results_df.head())