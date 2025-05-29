# filename: codebase/camb_power_spectrum.py
r"""
Compute the linear matter power spectrum P(k) at z=0 for a flat Lambda CDM cosmology using CAMB.

Parameters:
    H0: Hubble constant [km/s/Mpc]
    ombh2: Baryon density parameter [dimensionless]
    omch2: Cold dark matter density parameter [dimensionless]
    mnu: Sum of neutrino masses [eV]
    omk: Curvature parameter [dimensionless]
    tau: Optical depth to reionization [dimensionless]
    As: Scalar amplitude [dimensionless]
    ns: Scalar spectral index [dimensionless]
    kmax: Maximum k [1/Mpc]
    kmin: Minimum k [1/Mpc]
    num_k: Number of k points [dimensionless]

Outputs:
    Saves data/result.csv with columns:
        kh: Wavenumber [h/Mpc]
        P_k: Linear matter power spectrum [(Mpc/h)^3]
    Prints a summary of the results to the console.
"""

import numpy as np
import pandas as pd
import os

import camb
from camb import model, initialpower

# Ensure data directory exists
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Cosmological parameters
H0 = 67.5  # Hubble constant [km/s/Mpc]
ombh2 = 0.022  # Omega_b h^2 [dimensionless]
omch2 = 0.122  # Omega_c h^2 [dimensionless]
mnu = 0.06  # Sum of neutrino masses [eV]
omk = 0.0  # Curvature [dimensionless]
tau = 0.06  # Optical depth [dimensionless]
As = 2e-9  # Scalar amplitude [dimensionless]
ns = 0.965  # Scalar spectral index [dimensionless]
kmax = 2.0  # Maximum k [1/Mpc]
kmin = 1e-4  # Minimum k [1/Mpc]
num_k = 200  # Number of k points

# Generate k array in h/Mpc
kh_array = np.linspace(kmin, 1.0, num_k)  # [h/Mpc]

# Set up CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns)
pars.set_matter_power(redshifts=[0.0], kmax=kmax)
pars.NonLinear = model.NonLinear_none  # Linear power spectrum

# Calculate results
results = camb.get_results(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints = 1000)

# Interpolate P(k) at desired kh_array
from scipy.interpolate import interp1d
pk_interp = interp1d(kh, pk[0], kind='cubic', bounds_error=False, fill_value="extrapolate")
P_k = pk_interp(kh_array)  # (Mpc/h)^3

# Save to CSV
df = pd.DataFrame({'kh': kh_array, 'P_k': P_k})
csv_path = os.path.join(output_dir, "result.csv")
df.to_csv(csv_path, index=False)

# Print summary
np.set_printoptions(precision=6, suppress=True)
print("Linear matter power spectrum P(k) at z=0 computed for 200 k values.")
print("Cosmological parameters:")
print("  H0 = " + str(H0) + " km/s/Mpc")
print("  Omega_b h^2 = " + str(ombh2))
print("  Omega_c h^2 = " + str(omch2))
print("  Sum m_nu = " + str(mnu) + " eV")
print("  Omega_k = " + str(omk))
print("  tau = " + str(tau))
print("  As = " + str(As))
print("  ns = " + str(ns))
print("k range: " + str(kmin) + " < kh < 1 (h/Mpc), 200 points")
print("First 5 rows of result.csv:")
print(df.head(5).to_string(index=False))
print("Last 5 rows of result.csv:")
print(df.tail(5).to_string(index=False))
print("Results saved to " + csv_path)