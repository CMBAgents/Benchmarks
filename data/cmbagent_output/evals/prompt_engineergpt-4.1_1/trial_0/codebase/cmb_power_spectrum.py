# filename: codebase/cmb_power_spectrum.py
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
ombh2 = 0.022  # Omega_b h^2
omch2 = 0.122  # Omega_c h^2
mnu = 0.06  # sum m_nu [eV]
omk = 0.05  # Omega_k (curvature)
tau = 0.06  # optical depth
As = 2e-9  # scalar amplitude
ns = 0.965  # scalar spectral index

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

# l values (CAMB returns l=0,1,...)
ell = np.arange(totCL.shape[0])  # l=0,1,...,lmax

# Compute l(l+1)C_l^{TT}/(2pi) for l=2 to lmax
lvals = np.arange(lmin, lmax+1)
cltt = totCL[lmin:lmax+1, 0]  # TT spectrum, units: muK^2

# Compute l(l+1)C_l/(2pi)
factor = lvals * (lvals + 1) / (2.0 * np.pi)
cltt_scaled = factor * cltt  # units: muK^2

# Save to CSV
df = pd.DataFrame({'l': lvals, 'TT': cltt_scaled})
csv_path = os.path.join(output_dir, "result.csv")
df.to_csv(csv_path, index=False)

# Print summary of results
np.set_printoptions(precision=6, suppress=True)
pd.set_option('display.float_format', lambda x: '%.6g' % x)
print("CMB temperature power spectrum (l(l+1)C_l^{TT}/(2pi)) in microK^2, l=2 to l=3000")
print("First 5 rows:\n" + str(df.head()))
print("Last 5 rows:\n" + str(df.tail()))
print("Results saved to " + csv_path)