# filename: codebase/calc_ang_diameter.py
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

# Calculate results for the specified redshift range
results = camb.get_results(pars)

# Generate 100 evenly spaced redshift points from 0 to 4
z_values = np.linspace(0, 4, 100)

# Calculate angular diameter distance for each redshift
 d_A_values = results.angular_diameter_distance(z_values)

# Create a DataFrame to store the results
df = pd.DataFrame({'z': z_values, 'd_A': d_A_values})

# Save the results to a CSV file
df.to_csv('data/result.csv', index=False)

# Print the results to the console
print("Angular diameter distances calculated for redshifts from 0 to 4.")
print(df)