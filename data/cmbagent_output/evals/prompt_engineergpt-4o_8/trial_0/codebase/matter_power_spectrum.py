# filename: codebase/matter_power_spectrum.py
import camb
from camb import model, initialpower
import numpy as np
import pandas as pd

# Set cosmological parameters
H0 = 67.5  # Hubble constant in km/s/Mpc
ombh2 = 0.022  # Baryon density parameter
omch2 = 0.122  # Cold dark matter density parameter
mnu = 0.06  # Neutrino mass sum in eV
omk = 0.0  # Curvature parameter
tau = 0.06  # Optical depth to reionization
As = 2e-9  # Scalar amplitude
ns = 0.965  # Scalar spectral index

# Initialize CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns)
pars.set_matter_power(redshifts=[0], kmax=2.0)

# Get results from CAMB
results = camb.get_results(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints=200)

# Save results to CSV
data = {'k': kh, 'P_k': pk[0]}
df = pd.DataFrame(data)
df.to_csv('data/result.csv', index=False)

# Print results
print("Linear matter power spectrum calculated and saved to 'data/result.csv'.")
print("First few rows of the result:")
print(df.head())