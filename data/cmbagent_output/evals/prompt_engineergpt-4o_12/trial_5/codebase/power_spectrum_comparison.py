# filename: codebase/power_spectrum_comparison.py
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
sum_mnu = 0.11  # Neutrino mass sum in eV

# Define k range
k_values = np.linspace(1e-4, 2, 200)  # 200 evenly spaced k values

# Function to calculate power spectrum for a given neutrino hierarchy
def calculate_power_spectrum(neutrino_hierarchy):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=sum_mnu, 
                       omk=0, neutrino_hierarchy=neutrino_hierarchy)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=[0], kmax=2.0)
    
    # Calculate results for these parameters
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=2, npoints=200)
    return pk[0]  # Return power spectrum at z=0

# Calculate power spectra for normal and inverted hierarchies
pk_normal = calculate_power_spectrum('normal')
pk_inverted = calculate_power_spectrum('inverted')

# Calculate relative difference
rel_diff = (pk_inverted / pk_normal) - 1

# Save results to CSV
df = pd.DataFrame({'k': k_values, 'rel_diff': rel_diff})
df.to_csv('data/result.csv', index=False)

# Print detailed results
print("Relative difference in the linear matter power spectrum calculated and saved to 'data/result.csv'.")
print("First few entries of the result:")
print(df.head())