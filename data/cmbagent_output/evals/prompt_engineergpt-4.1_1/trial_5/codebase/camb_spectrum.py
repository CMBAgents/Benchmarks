# filename: codebase/camb_spectrum.py
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
    print("CAMB is not installed. Please install camb and rerun the code.")
    raise e

# Cosmological parameters
H0 = 67.3  # Hubble constant [km/s/Mpc]
ombh2 = 0.022  # Omega_b h^2 [unitless]
omch2 = 0.122  # Omega_c h^2 [unitless]
mnu = 0.06  # Sum of neutrino masses [eV]
omk = 0.05  # Curvature [unitless]
tau = 0.06  # Optical depth [unitless]
As = 2e-9  # Scalar amplitude [unitless]
ns = 0.965  # Scalar spectral index [unitless]

# Set up CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns)
pars.set_for_lmax(3000, lens_potential_accuracy=0)

# Calculate results
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000)
totCL = powers['total']

# l values: CAMB returns from l=0, so index matches l
lmin = 2
lmax = 3000
ls = np.arange(lmin, lmax + 1)
# totCL shape: (lmax+1, 4), columns: TT, EE, BB, TE

# Compute l(l+1)C_l^{TT}/(2pi) in microK^2
TT = totCL[lmin:lmax+1, 0]  # TT spectrum, units: microK^2
TT_power = ls * (ls + 1) * TT / (2 * np.pi)  # [microK^2]

# Save to CSV
df = pd.DataFrame({'l': ls, 'TT': TT_power})
csv_path = os.path.join("data", "result.csv")
df.to_csv(csv_path, index=False)

# Print summary
np.set_printoptions(precision=4, suppress=True, linewidth=120)
print("CMB temperature power spectrum (l(l+1)C_l^{TT}/(2pi)) [microK^2] computed for non-flat Lambda CDM cosmology.")
print("Parameters used:")
print("  H0 = " + str(H0) + " km/s/Mpc")
print("  Omega_b h^2 = " + str(ombh2))
print("  Omega_c h^2 = " + str(omch2))
print("  Sum m_nu = " + str(mnu) + " eV")
print("  Omega_k = " + str(omk))
print("  tau = " + str(tau))
print("  As = " + str(As))
print("  ns = " + str(ns))
print("Results saved to: " + csv_path)
print("First 10 rows of the result:")
print(df.head(10).to_string(index=False))
print("Last 10 rows of the result:")
print(df.tail(10).to_string(index=False))
