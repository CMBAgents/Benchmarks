# filename: codebase/camb_power_spectrum.py
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

# Cosmological parameters
H0 = 70.0  # Hubble constant [km/s/Mpc]
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
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
totCL = powers['total']  # Shape: (lmax+1, 4), columns: TT, EE, BB, TE

# Extract l and TT
ell = np.arange(totCL.shape[0])  # l = 0, 1, ..., lmax
TT = totCL[:, 0]  # TT spectrum in muK^2

# Restrict to l=2 to l=3000
mask = (ell >= lmin) & (ell <= lmax)
ell = ell[mask]
TT = TT[mask]

# Save to CSV
df = pd.DataFrame({'l': ell, 'TT': TT})
csv_path = os.path.join("data", "result.csv")
df.to_csv(csv_path, index=False)

# Print summary
np.set_printoptions(precision=6, suppress=True, linewidth=120)
print("CMB temperature power spectrum (C_l^{TT}) computed for flat Lambda CDM cosmology.")
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
print("First 10 rows of the result:")
print(df.head(10).to_string(index=False))
print("Last 10 rows of the result:")
print(df.tail(10).to_string(index=False))
