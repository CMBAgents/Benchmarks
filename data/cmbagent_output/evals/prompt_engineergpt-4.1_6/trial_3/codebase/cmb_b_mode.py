# filename: codebase/cmb_b_mode.py
import os
import numpy as np
import pandas as pd

# Ensure data directory exists
database_path = "data"
if not os.path.exists(database_path):
    os.makedirs(database_path)

try:
    import camb
    from camb import model, initialpower
except ImportError as e:
    print("CAMB package is required but not installed. Please install camb and rerun the code.")
    raise e

# Cosmological parameters
H0 = 67.5  # Hubble constant [km/s/Mpc]
ombh2 = 0.022  # Omega_b h^2
omch2 = 0.122  # Omega_c h^2
mnu = 0.06  # sum m_nu [eV]
omk = 0.0  # Omega_k
tau = 0.06  # optical depth
r = 0.1  # tensor-to-scalar ratio
As = 2e-9  # scalar amplitude
ns = 0.965  # scalar spectral index

# Set up CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns, r=r)
pars.WantTensors = True
pars.set_for_lmax(3000, lens_potential_accuracy=0)

# Get results and power spectra
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
# 'raw_cl=True' returns Cl, not Dl, in units of microK^2

# Extract BB spectrum
cl = powers['total']  # columns: TT, EE, BB, TE, [PP, TP] if lensing
# cl shape: (lmax+1, ncols)
# BB is column 2 (index 2)
lmax = 3000
l_arr = np.arange(cl.shape[0])  # l = 0, 1, ..., lmax
BB = cl[:, 2]  # BB spectrum in microK^2

# Select l=2 to l=3000
lmin = 2
lmax = 3000
l_vals = np.arange(lmin, lmax + 1)
BB_vals = BB[lmin:lmax + 1]

# Save to CSV
df = pd.DataFrame({'l': l_vals, 'BB': BB_vals})
csv_path = os.path.join(database_path, "result.csv")
df.to_csv(csv_path, index=False)

# Print summary of results
pd.set_option("display.precision", 6)
pd.set_option("display.width", 120)
print("CMB B-mode polarization power spectrum (C_l^{BB}) computed for l=2 to l=3000.")
print("Results saved to " + csv_path)
print("First 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())
