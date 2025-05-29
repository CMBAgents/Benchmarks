# filename: codebase/camb_cmb_tt_spectrum.py
import os
import numpy as np
import pandas as pd
import camb

# Ensure output directory exists
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Cosmological parameters
H0 = 74.0  # Hubble constant [km/s/Mpc]
ombh2 = 0.022  # Omega_b h^2
omch2 = 0.122  # Omega_c h^2
mnu = 0.06  # Sum of neutrino masses [eV]
omk = 0.0  # Curvature
tau = 0.06  # Optical depth to reionization
As = 2e-9  # Scalar amplitude
ns = 0.965  # Scalar spectral index

# Multipole range
lmin = 2
lmax = 3000

# Set up CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns)
pars.set_for_lmax(lmax, lens_potential_accuracy=0)

# Calculate results
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
cl = powers['unlensed_scalar']

# Extract TT spectrum for l=2 to l=3000
# cl shape: (lmax+1, 4), columns: TT, EE, BB, TE
ls = np.arange(cl.shape[0])
mask = (ls >= lmin) & (ls <= lmax)
ls = ls[mask]
TT = cl[mask, 0]  # TT spectrum in μK^2

# Save to CSV
df = pd.DataFrame({'l': ls, 'TT': TT})
csv_path = os.path.join(output_dir, "result.csv")
df.to_csv(csv_path, index=False)

# Print summary
np.set_printoptions(precision=6, suppress=True)
print("CMB TT power spectrum computed for flat Lambda CDM cosmology.")
print("Parameters:")
print("  H0 = " + str(H0) + " km/s/Mpc")
print("  Omega_b h^2 = " + str(ombh2))
print("  Omega_c h^2 = " + str(omch2))
print("  Sum m_nu = " + str(mnu) + " eV")
print("  Omega_k = " + str(omk))
print("  tau = " + str(tau))
print("  As = " + str(As))
print("  ns = " + str(ns))
print("Multipole range: l = " + str(lmin) + " to " + str(lmax))
print("Results saved to: " + csv_path)
print("First 5 rows of the result:")
print(df.head())
print("Last 5 rows of the result:")
print(df.tail())
