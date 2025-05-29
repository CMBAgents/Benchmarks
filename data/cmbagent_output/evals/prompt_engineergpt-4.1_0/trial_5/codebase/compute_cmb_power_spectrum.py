# filename: codebase/compute_cmb_power_spectrum.py
import os
import numpy as np
import pandas as pd

# Ensure data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

try:
    import camb
    from camb import model, initialpower
except ImportError as e:
    print("CAMB package is required but not installed. Please install camb and rerun the code.")
    raise e

# Cosmological parameters (units in comments)
H0 = 67.5  # Hubble constant [km/s/Mpc]
ombh2 = 0.02  # Omega_b h^2 (baryon density)
omch2 = 0.122  # Omega_c h^2 (cold dark matter density)
mnu = 0.06  # Sum of neutrino masses [eV]
omk = 0.0  # Curvature (flat universe)
tau = 0.06  # Optical depth to reionization
As = 2e-9  # Scalar amplitude
ns = 0.965  # Scalar spectral index
lmax = 3000  # Maximum multipole

# Set up CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns)
pars.set_for_lmax(lmax, lens_potential_accuracy=0)

# Calculate results
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax)
totCL = powers['total']

# l values (CAMB returns l from 0)
ell = np.arange(totCL.shape[0])  # l = 0, 1, ..., lmax

# Compute l(l+1)C_l^{TT}/(2pi) in muK^2 for l=2 to l=3000
lmin = 2
lmax_out = 3000
lvals = np.arange(lmin, lmax_out + 1)
cltt = totCL[lmin:lmax_out + 1, 0]  # TT spectrum, units: muK^2

# Compute l(l+1)C_l/(2pi)
factor = lvals * (lvals + 1) / (2.0 * np.pi)
cltt_scaled = factor * cltt  # units: muK^2

# Save to CSV
df = pd.DataFrame({'l': lvals, 'TT': cltt_scaled})
output_path = os.path.join("data", "result.csv")
df.to_csv(output_path, index=False)

# Print summary
np.set_printoptions(precision=4, suppress=True, linewidth=120)
print("CMB temperature power spectrum (l(l+1)C_l^{TT}/(2pi)) computed for l=2 to l=3000.")
print("Results saved to data/result.csv")
print("First 10 rows:")
print(df.head(10).to_string(index=False))
print("\nLast 10 rows:")
print(df.tail(10).to_string(index=False))
print("\nUnits: TT in microkelvin^2 (uK^2), l is dimensionless.")