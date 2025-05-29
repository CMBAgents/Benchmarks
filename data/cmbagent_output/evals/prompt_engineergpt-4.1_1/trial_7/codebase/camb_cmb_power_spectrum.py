# filename: codebase/camb_cmb_power_spectrum.py
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
H0 = 67.3  # Hubble constant [km/s/Mpc]
ombh2 = 0.022  # Omega_b h^2 [dimensionless]
omch2 = 0.122  # Omega_c h^2 [dimensionless]
mnu = 0.06  # Sum of neutrino masses [eV]
omk = 0.05  # Omega_k [dimensionless]
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
pars.WantTensors = False
pars.Want_CMB = True

# Calculate results
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
totCL = powers['total']

# l values
ell = np.arange(totCL.shape[0])  # ell[0]=0, ell[1]=1, ...
# Compute l(l+1)C_l/(2pi) for TT, in μK^2
cl_TT = totCL[:, 0]  # TT spectrum, units: μK^2

# Only keep l=2 to lmax
l_vals = ell[lmin:lmax+1]
cl_TT_vals = cl_TT[lmin:lmax+1]
cl_TT_scaled = l_vals * (l_vals + 1) * cl_TT_vals / (2.0 * np.pi)  # μK^2

# Save to CSV
df = pd.DataFrame({'l': l_vals, 'TT': cl_TT_scaled})
csv_path = os.path.join(output_dir, "result.csv")
df.to_csv(csv_path, index=False)

# Print results to console in full
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: "%.8e" % x)
print("CMB temperature power spectrum (l(l+1)C_l^{TT}/(2pi)) in μK^2 for l=2 to l=3000:")
print(df)

print("\nResults saved to " + csv_path)