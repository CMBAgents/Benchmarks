# filename: codebase/delensed_cmb_bb.py
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
ombh2 = 0.022  # Baryon density Omega_b h^2
omch2 = 0.122  # Cold dark matter density Omega_c h^2
mnu = 0.06  # Neutrino mass sum [eV]
omk = 0.0  # Curvature Omega_k
tau = 0.06  # Optical depth to reionization
r = 0.1  # Tensor-to-scalar ratio
As = 2e-9  # Scalar amplitude
ns = 0.965  # Scalar spectral index

# Multipole range
lmin = 2
lmax = 3000

# Delensing efficiency (fraction of lensing B-modes removed)
delensing_efficiency = 0.10  # 10%

# Set up CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns, r=r)
pars.WantTensors = True
pars.set_for_lmax(lmax, lens_potential_accuracy=1)
pars.Want_CMB_lensing = True

# Calculate results
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)

# Extract total BB power spectrum (includes lensing and primordial)
cl_total = powers['total']  # shape: (lmax+1, 4), columns: TT, EE, BB, TE

# Extract lensing-only BB power spectrum
cl_lensed = powers['lensed_scalar']  # shape: (lmax+1, 4)
cl_unlensed = powers['unlensed_scalar']  # shape: (lmax+1, 4)

# Compute lensing B-mode: lensing_BB = lensed_scalar_BB - unlensed_scalar_BB
lensing_BB = cl_lensed[:,2] - cl_unlensed[:,2]  # [muK^2]

# Apply delensing: reduce lensing B-modes by delensing_efficiency
# Delensed BB = primordial_BB + (1 - delensing_efficiency) * lensing_BB
# primordial_BB = total_BB - lensing_BB
primordial_BB = cl_total[:,2] - lensing_BB
delensed_BB = primordial_BB + (1.0 - delensing_efficiency) * lensing_BB

# Prepare output for l=2 to l=3000
ells = np.arange(lmin, lmax+1)
BB = delensed_BB[lmin:lmax+1]  # [muK^2]

# Save to CSV
df = pd.DataFrame({'l': ells, 'BB': BB})
csv_path = os.path.join(output_dir, "result.csv")
df.to_csv(csv_path, index=False)

# Print summary of results
np.set_printoptions(precision=6, suppress=True)
print("Delensed CMB B-mode power spectrum (C_ell^{BB}) saved to " + csv_path)
print("Columns: l (multipole), BB (delensed C_ell^{BB} in micro-Kelvin^2)")
print("First 10 rows:")
print(df.head(10).to_string(index=False))
print("\nLast 10 rows:")
print(df.tail(10).to_string(index=False))
print("\nParameter summary:")
print("H0 = " + str(H0) + " km/s/Mpc, ombh2 = " + str(ombh2) + ", omch2 = " + str(omch2) + ", mnu = " + str(mnu) + " eV")
print("omk = " + str(omk) + ", tau = " + str(tau) + ", r = " + str(r) + ", As = " + str(As) + ", ns = " + str(ns))
print("Delensing efficiency: " + str(delensing_efficiency*100) + "% (lensing B-modes reduced by 10%)")