# filename: codebase/compute_cmb_power_spectrum.py
import camb
from camb import model, initialpower
import numpy as np
import pandas as pd

# Define cosmological parameters
H0 = 67.5
ombh2 = 0.022
omch2 = 0.122
omk = -0.01  # non-flat universe
tau = 0.06
As = 2e-9
ns = 0.965

# Set up the CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns)

# Set the maximum multipole
pars.set_for_lmax(2500, lens_potential_accuracy=0)

# Calculate results for the CMB power spectra
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=2500, raw_cl=True)[0]

# Get multipole moments
ells = np.arange(powers.shape[0])

# Convert results to a DataFrame and save to CSV
df = pd.DataFrame(powers, columns=["TT", "EE", "BB", "TE"])
df.insert(0, "ell", ells)
df.to_csv("data/result.csv", index=False)

# Print first and last 10 rows of the result
print("First 10 rows:")
print(df.head(10))
print("\nLast 10 rows:")
print(df.tail(10))

# Print cosmological parameters
print("\nCosmological Parameters:")
print("H0 = " + str(H0))
print("ombh2 = " + str(ombh2))
print("omch2 = " + str(omch2))
print("omk = " + str(omk))
print("tau = " + str(tau))
print("As = " + str(As))
print("ns = " + str(ns))
