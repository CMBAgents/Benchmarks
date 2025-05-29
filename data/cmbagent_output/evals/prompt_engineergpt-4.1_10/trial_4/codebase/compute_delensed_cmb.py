# filename: codebase/compute_delensed_cmb.py
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower
import os

# Output directory
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Cosmological parameters
H0 = 67.5  # Hubble constant [km/s/Mpc]
ombh2 = 0.022  # Baryon density [Ω_b h^2]
omch2 = 0.122  # Cold dark matter density [Ω_c h^2]
mnu = 0.06  # Neutrino mass sum [eV]
omk = 0.0  # Curvature [Ω_k]
tau = 0.06  # Optical depth to reionization
As = 2e-9  # Scalar amplitude
ns = 0.965  # Scalar spectral index

# Delensing efficiency (fraction of lensing removed)
delensing_efficiency = 0.8

# Multipole range
lmin = 2
lmax = 3000

# Set up CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns)
pars.set_for_lmax(lmax, lens_potential_accuracy=1)
pars.WantTensors = False

# Get results for both lensed and unlensed spectra
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total', 'unlensed_scalar'])

# Extract lensed and unlensed TT spectra
cl_lensed = powers['total'][:, 0]  # TT, lensed, units: μK^2
cl_unlensed = powers['unlensed_scalar'][:, 0]  # TT, unlensed, units: μK^2

# Compute delensed spectrum: C_l^delensed = C_l^unlensed + (1 - delensing_efficiency) * (C_l^lensed - C_l^unlensed)
cl_delensed = cl_unlensed + (1.0 - delensing_efficiency) * (cl_lensed - cl_unlensed)

# Compute l(l+1)C_l/(2π) for l=2 to lmax
ells = np.arange(lmin, lmax + 1)
factor = ells * (ells + 1) / (2.0 * np.pi)
# cl_delensed[ells] is already in μK^2
cl_delensed_plot = factor * cl_delensed[ells]

# Save to CSV
df = pd.DataFrame({'l': ells, 'TT': cl_delensed_plot})
csv_path = os.path.join(output_dir, "result.csv")
df.to_csv(csv_path, index=False)

# Print summary
np.set_printoptions(precision=6, suppress=True, linewidth=120)
print("Delensed CMB temperature power spectrum (l(l+1)C_l^{TT}/(2π)) computed for l=2 to l=3000.")
print("Cosmological parameters:")
print("  H0 = " + str(H0) + " km/s/Mpc")
print("  Ω_b h^2 = " + str(ombh2))
print("  Ω_c h^2 = " + str(omch2))
print("  Σm_ν = " + str(mnu) + " eV")
print("  Ω_k = " + str(omk))
print("  τ = " + str(tau))
print("  A_s = " + str(As))
print("  n_s = " + str(ns))
print("Delensing efficiency: " + str(delensing_efficiency * 100.0) + "%")
print("Results saved to: " + csv_path)
print("First 10 rows of the result:")
print(df.head(10).to_string(index=False))
print("Last 10 rows of the result:")
print(df.tail(10).to_string(index=False))
