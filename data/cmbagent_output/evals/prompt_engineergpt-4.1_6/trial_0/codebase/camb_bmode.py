# filename: codebase/camb_bmode.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

# Ensure output directory exists
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
# 'raw_cl=True' gives C_l, not D_l

# Extract the total power spectrum (includes tensor modes)
cl = powers['total']  # shape: (lmax+1, 4) for TT, EE, BB, TE

# l values: cl[ell] corresponds to ell=l, for l=0 to lmax
# We want l=2 to l=3000
ells = np.arange(cl.shape[0])  # l=0 to lmax
mask = (ells >= lmin) & (ells <= lmax)
ells = ells[mask]
cl_bb = cl[mask, 2]  # BB is the third column (index 2), units: muK^2

# Save to CSV
df = pd.DataFrame({'l': ells, 'BB': cl_bb})
csv_path = os.path.join(output_dir, "result.csv")
df.to_csv(csv_path, index=False)

# Print summary
pd.set_option("display.precision", 6)
pd.set_option("display.max_rows", 10)
print("CMB B-mode polarization power spectrum (C_l^{BB}) computed for l=2 to l=3000.")
print("Units: BB in microK^2.")
print("First and last 5 rows of the result:")
print(pd.concat([df.head(5), df.tail(5)]))
print("\nFull results saved to: " + csv_path)