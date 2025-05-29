# filename: codebase/relative_power_spectrum.py
import camb
from camb import model, initialpower
import numpy as np
import pandas as pd

# Set cosmological parameters
H0 = 67.5  # Hubble constant in km/s/Mpc
ombh2 = 0.022  # Baryon density
omch2 = 0.122  # Cold dark matter density
As = 2e-9  # Scalar amplitude
ns = 0.965  # Scalar spectral index
mnu = 0.11  # Neutrino mass sum in eV

# Define k range
tk_values = np.linspace(1e-4, 2, 200)  # 200 evenly spaced k values


def get_matter_power_spectrum(neutrino_hierarchy):
    """
    Calculate the linear matter power spectrum for a given neutrino hierarchy.
    
    Parameters:
    neutrino_hierarchy (str): 'normal' or 'inverted' neutrino hierarchy.
    
    Returns:
    np.ndarray: Linear matter power spectrum P(k) in (Mpc/h)^3.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=0, neutrino_hierarchy=neutrino_hierarchy)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=[0], kmax=2.0)
    
    # Calculate results for these parameters
    results = camb.get_results(pars)
    
    # Get the matter power spectrum
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=2.0, npoints=200)
    return pk[0]  # Return P(k) at z=0


# Calculate power spectra for both hierarchies
pk_normal = get_matter_power_spectrum('normal')
pk_inverted = get_matter_power_spectrum('inverted')

# Calculate relative difference
rel_diff = (pk_inverted / pk_normal) - 1

# Save results to CSV
results_df = pd.DataFrame({'k': tk_values, 'rel_diff': rel_diff})
results_df.to_csv('data/result.csv', index=False)

print("Relative difference in the linear matter power spectrum calculated and saved to 'data/result.csv'.")