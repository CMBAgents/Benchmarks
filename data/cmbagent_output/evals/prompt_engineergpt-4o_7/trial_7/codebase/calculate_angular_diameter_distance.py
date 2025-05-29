# filename: codebase/calculate_angular_diameter_distance.py
import numpy as np
import camb
from camb import model, initialpower
import pandas as pd

# Set cosmological parameters
H0 = 67.5  # Hubble constant in km/s/Mpc
ombh2 = 0.022  # Baryon density
omch2 = 0.122  # Cold dark matter density
mnu = 0.06  # Neutrino mass sum in eV
omk = 0  # Curvature
tau = 0.06  # Optical depth to reionization
As = 2e-9  # Scalar amplitude
ns = 0.965  # Scalar spectral index

# Initialize CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns)
pars.set_for_lmax(2500, lens_potential_accuracy=0)

# Calculate results for these parameters
results = camb.get_results(pars)

# Define redshift range
z_values = np.linspace(0, 4, 100)

# Calculate angular diameter distance
d_A_values = results.angular_diameter_distance(z_values)

# Save results to CSV
data = pd.DataFrame({'z': z_values, 'd_A': d_A_values})
data.to_csv('data/result.csv', index=False)

# Print results
print("Angular diameter distances calculated and saved to 'data/result.csv'.")
print(data)