# filename: codebase/compute_cmb_bb.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

# Ensure data directory exists
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Cosmological parameters (units in comments)
H0 = 67.5  # Hubble constant [km/s/Mpc]
ombh2 = 0.022  # Omega_b h^2 [dimensionless]
omch2 = 0.122  # Omega_c h^2 [dimensionless]
mnu = 0.06  # sum m_nu [eV]
omk = 0.0  # Omega_k [dimensionless]
tau = 0.06  # optical depth [dimensionless]
r = 0.1  # tensor-to-scalar ratio [dimensionless]
As = 2e-9  # scalar amplitude [dimensionless]
ns = 0.965  # scalar spectral index [dimensionless]

# Set up CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns, r=r)
pars.WantTensors = True
pars.set_for_lmax(3000, lens_potential_accuracy=0)

# Calculate results
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
# powers['tensor'] has columns: TT, EE, BB, TE, (and possibly others)
tensor_cls = powers['tensor']  # shape: (lmax+1, ncols)

# l values: 0 to lmax
lmax = 3000
l = np.arange(tensor_cls.shape[0])  # l = 0, 1, ..., lmax

# Extract BB spectrum for l=2 to l=3000
l_out = l[2:lmax+1]  # l=2 to l=3000
BB = tensor_cls[2:lmax+1, 2]  # BB is the third column (index 2), units: muK^2

# Save to CSV
df = pd.DataFrame({'l': l_out, 'BB': BB})
csv_path = os.path.join(output_dir, "result.csv")
df.to_csv(csv_path, index=False)

# Print summary
np.set_printoptions(precision=6, suppress=True, linewidth=120)
print("CMB B-mode polarization power spectrum (C_l^{BB}) computed for l=2 to l=3000.")
print("Cosmological parameters used:")
print("  H0 = " + str(H0) + " km/s/Mpc")
print("  Omega_b h^2 = " + str(ombh2))
print("  Omega_c h^2 = " + str(omch2))
print("  Sum m_nu = " + str(mnu) + " eV")
print("  Omega_k = " + str(omk))
print("  tau = " + str(tau))
print("  r = " + str(r))
print("  As = " + str(As))
print("  ns = " + str(ns))
print("Results saved to: " + csv_path)
print("First 10 rows of the result:")
print(df.head(10).to_string(index=False))
print("Last 10 rows of the result:")
print(df.tail(10).to_string(index=False))
