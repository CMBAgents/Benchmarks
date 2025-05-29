# filename: codebase/camb_cmb_power_spectrum.py
import os
import numpy as np
import pandas as pd

# Ensure output directory exists
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    import camb
    from camb import model, initialpower
except ImportError as e:
    print("CAMB package is required but not installed. Please install 'camb' and rerun the code.")
    raise e

# Cosmological parameters
H0 = 67.3  # Hubble constant [km/s/Mpc]
ombh2 = 0.022  # Omega_b h^2 [dimensionless]
omch2 = 0.122  # Omega_c h^2 [dimensionless]
mnu = 0.06  # Sum of neutrino masses [eV]
omk = 0.05  # Curvature Omega_k [dimensionless]
tau = 0.06  # Optical depth to reionization [dimensionless]
As = 2e-9  # Scalar amplitude [dimensionless]
ns = 0.965  # Scalar spectral index [dimensionless]

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
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total'])

# Extract TT spectrum
totCL = powers['total']
# totCL shape: (lmax+1, 4), columns: TT, EE, BB, TE

# l values: CAMB returns from l=0 to l=lmax
ells = np.arange(totCL.shape[0])  # l=0,...,lmax

# Compute l(l+1)C_l^{TT}/(2pi) for l=2 to lmax
ells_out = np.arange(lmin, lmax+1)
cl_tt = totCL[lmin:lmax+1, 0]  # TT spectrum in muK^2

# Compute l(l+1)C_l/(2pi)
cl_tt_scaled = ells_out * (ells_out + 1) * cl_tt / (2.0 * np.pi)  # [muK^2]

# Save to CSV
df = pd.DataFrame({'l': ells_out, 'TT': cl_tt_scaled})
csv_path = os.path.join(output_dir, "result.csv")
df.to_csv(csv_path, index=False)

# Print summary of results
pd.set_option('display.float_format', lambda x: '%.6e' % x)
print("CMB temperature power spectrum (l(l+1)C_l^{TT}/(2pi)) in muK^2 saved to " + csv_path)
print("First 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())