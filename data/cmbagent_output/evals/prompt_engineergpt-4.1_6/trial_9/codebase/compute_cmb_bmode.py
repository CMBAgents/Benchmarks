# filename: codebase/compute_cmb_bmode.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

# Ensure data directory exists
database_path = "data"
if not os.path.exists(database_path):
    os.makedirs(database_path)

# Cosmological parameters
H0 = 67.5  # Hubble constant [km/s/Mpc]
ombh2 = 0.022  # Baryon density parameter [dimensionless]
omch2 = 0.122  # Cold dark matter density parameter [dimensionless]
mnu = 0.06  # Neutrino mass sum [eV]
omk = 0.0  # Curvature [dimensionless]
tau = 0.06  # Optical depth to reionization [dimensionless]
r = 0.1  # Tensor-to-scalar ratio [dimensionless]
As = 2e-9  # Scalar amplitude [dimensionless]
ns = 0.965  # Scalar spectral index [dimensionless]

# Multipole range
lmin = 2
lmax = 3000

# Set up CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns, r=r)
pars.WantTensors = True
pars.set_for_lmax(lmax, lens_potential_accuracy=0)

# Get results and power spectra
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)  # 'raw_cl=True' gives Cl, not Dl

# Extract the total power spectrum (includes tensor modes)
cl = powers['total']  # shape: (lmax+1, 4) for TT, EE, BB, TE

# l values: cl[2:] corresponds to l=2 to lmax
l_array = np.arange(lmin, lmax+1, dtype=int)
BB = cl[lmin:lmax+1, 2]  # BB spectrum in muK^2

# Save to CSV
output_file = os.path.join(database_path, "result.csv")
df = pd.DataFrame({'l': l_array, 'BB': BB})
df.to_csv(output_file, index=False)

# Print summary
np.set_printoptions(precision=6, suppress=True)
pd.set_option("display.precision", 6)
pd.set_option("display.max_rows", 10)
print("CMB B-mode polarization power spectrum (C_l^{BB}) computed for l=2 to l=3000.")
print("Units: BB in microK^2.")
print("First and last 5 rows of the result:")
print(pd.concat([df.head(5), df.tail(5)]))
print("Full results saved to " + output_file)