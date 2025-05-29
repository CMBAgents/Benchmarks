# filename: codebase/camb_bb_power_spectrum.py
import os
import numpy as np
import pandas as pd

# Set up output directory
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Import camb and set up parameters
import camb
from camb import model, initialpower

# Cosmological parameters
H0 = 67.5  # Hubble parameter [km/s/Mpc]
ombh2 = 0.022  # Baryon density parameter [dimensionless]
omch2 = 0.122  # Cold dark matter density parameter [dimensionless]
mnu = 0.06  # Neutrino mass sum [eV]
omk = 0.0  # Curvature [dimensionless]
tau = 0.06  # Optical depth to reionization [dimensionless]
r = 0.0  # Tensor-to-scalar ratio [dimensionless]
As = 2e-9  # Scalar amplitude [dimensionless]
ns = 0.965  # Scalar spectral index [dimensionless]

# Multipole range
lmin = 2
lmax = 3000

# Set up CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns, r=r)
pars.set_for_lmax(lmax, lens_potential_accuracy=1)
pars.WantTensors = True  # Needed for BB spectrum

# Run CAMB
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total'])

# Extract BB spectrum
cl = powers['total']  # shape: (lmax+1, 4) for TT, EE, BB, TE
# cl[:,2] is BB
ells = np.arange(cl.shape[0])  # 0,1,...,lmax
BB = cl[:,2]  # BB spectrum in muK^2

# Compute l(l+1)C_l^{BB}/(2pi) for l=2..lmax
l_arr = np.arange(lmin, lmax+1)
BB_arr = BB[lmin:lmax+1]  # Select l=2..lmax
BB_lensed = l_arr * (l_arr + 1) * BB_arr / (2.0 * np.pi)  # [muK^2]

# Save to CSV
df = pd.DataFrame({'l': l_arr, 'BB': BB_lensed})
csv_path = os.path.join(output_dir, "result.csv")
df.to_csv(csv_path, index=False)

# Print summary
pd.set_option("display.precision", 8)
pd.set_option("display.max_rows", 10)
print("CMB B-mode polarization power spectrum (l(l+1)C_l^{BB}/(2pi)) [muK^2] for l=2..3000")
print("Cosmological parameters:")
print("  H0 = " + str(H0) + " km/s/Mpc")
print("  ombh2 = " + str(ombh2))
print("  omch2 = " + str(omch2))
print("  mnu = " + str(mnu) + " eV")
print("  omk = " + str(omk))
print("  tau = " + str(tau))
print("  r = " + str(r))
print("  As = " + str(As))
print("  ns = " + str(ns))
print("\nFirst and last 5 rows of the result:")
print(df.head(5))
print("...")
print(df.tail(5))
print("\nResults saved to " + csv_path)