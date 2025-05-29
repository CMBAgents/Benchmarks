# filename: codebase/calculate_relative_pk.py
import camb
from camb import model
import numpy as np
import pandas as pd
import os

# Define cosmological parameters
H0 = 67.5  # Hubble constant in km/s/Mpc
ombh2 = 0.022  # Baryon density * h^2
omch2 = 0.122  # Cold dark matter density * h^2
sum_mnu = 0.11  # Sum of neutrino masses in eV
As = 2.0e-9  # Scalar amplitude
ns = 0.965  # Scalar spectral index
omk = 0.0  # Curvature density (0 for flat)
# tau = 0.055 is CAMB's default in set_cosmology if not specified.

# Derived Hubble parameter h (dimensionless)
h_param = H0 / 100.0

# k-range definition for the output CSV column 'k' (which is k/h)
# The problem states "200 evenly spaced k values in the range 10^-4 < k h < 2 (Mpc^-1)"
# Let k_csv be the values in the CSV column 'k' (units: h/Mpc).
# Then physical k, k_phys = k_csv * h_param (units: Mpc^-1).
# The range constraint is on k_phys: 1e-4 Mpc^-1 < k_phys < 2.0 Mpc^-1.
# So, for k_csv: (1e-4 / h_param) h/Mpc < k_csv < (2.0 / h_param) h/Mpc.
min_k_csv = 1e-4 / h_param  # Min k/h value (units: h/Mpc)
max_k_csv = 2.0 / h_param   # Max k/h value (units: h/Mpc)
npoints = 200               # Number of k points

# Linearly spaced k_csv values (these are k/h in CAMB terminology)
kh_target_values = np.linspace(min_k_csv, max_k_csv, npoints) # units: h/Mpc

# Max k/h value for the interpolator's internal spline grid (must cover kh_target_values)
max_kh_for_interpolator_spline = kh_target_values[-1]

def get_pk(hierarchy_type, kh_input_values, max_kh_spline):
    r"""
    Calculates the linear matter power spectrum P(k/h) for a given neutrino hierarchy.

    Args:
        hierarchy_type (str): 'normal' or 'inverted'.
        kh_input_values (numpy.ndarray): Array of k/h values (units: h/Mpc) 
                                         at which to calculate P(k).
        max_kh_spline (float): Maximum k/h value (units: h/Mpc) for CAMB's 
                               internal B-spline generation for P(k).

    Returns:
        numpy.ndarray: Array of P(k/h) values (units: (Mpc/h)^3).
    """
    pars = camb.CAMBparams()
    
    # Set cosmological parameters
    # tau default in set_cosmology is 0.055.
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=sum_mnu, omk=omk)
    
    # Set initial power spectrum parameters
    # Default k_pivot for As and ns is 0.05 Mpc^-1
    pars.InitPower.set_params(As=As, ns=ns) 
    
    # Configure CAMB for linear matter power spectrum calculation
    pars.WantCls = False  # We do not need CMB angular power spectra (C_ls)
    pars.WantTransfer = True  # Essential for using get_matter_power_interpolator
    pars.NonLinear = model.NonLinear_none  # Calculate linear power spectrum
    
    # Set neutrino properties
    pars.num_massive_neutrinos = 3 # Number of massive neutrino species
    if hierarchy_type == 'normal':
        pars.nu_mass_eigenstates = 1 # Normal hierarchy
    elif hierarchy_type == 'inverted':
        pars.nu_mass_eigenstates = 2 # Inverted hierarchy
    else:
        raise ValueError("hierarchy_type must be 'normal' or 'inverted'")

    # Get the matter power interpolator object
    # This interpolator will provide P(k/h) in (Mpc/h)^3 for k/h inputs.
    PK_interpolator = camb.get_matter_power_interpolator(
        pars,
        nonlinear=False,  # We want the linear power spectrum
        redshifts=[0.0],  # Calculate P(k) at redshift z=0
        k_hunit=True,     # Input k values to P(z,k) are k/h (units: h/Mpc)
        hubble_units=True,# Output P(k) is in (Mpc/h)^3
        kmax=max_kh_spline, # Max k/h for the B-spline grid that is interpolated
        log_interp=True   # Interpolate log(P(k)) vs log(k) for better accuracy
    )

    # Interpolate P(k/h) at the target k/h values for z=0
    # pk_values are P(k/h) in (Mpc/h)^3
    pk_values = PK_interpolator.P(0.0, kh_input_values) # P(z, k_input) 
    
    return pk_values

# Create 'data' directory if it doesn't exist
# This ensures the directory for saving the CSV file is available.
os.makedirs('data', exist_ok=True)

# Calculate P(k) for normal hierarchy
print("Calculating P(k) for normal hierarchy...")
pk_normal = get_pk('normal', kh_target_values, max_kh_for_interpolator_spline)

# Calculate P(k) for inverted hierarchy
print("Calculating P(k) for inverted hierarchy...")
pk_inverted = get_pk('inverted', kh_target_values, max_kh_for_interpolator_spline)

# Calculate relative difference in P(k)
# Relative difference = (P(k)_inverted / P(k)_normal - 1)
relative_difference = (pk_inverted / pk_normal) - 1.0

# Create pandas DataFrame to store the results
df_results = pd.DataFrame({
    'k': kh_target_values,          # k is k/h, units: h/Mpc
    'rel_diff': relative_difference # Relative difference, dimensionless
})

# Save results to a CSV file in the 'data' directory
csv_filename = os.path.join('data', 'result.csv')
# Use scientific notation for float formatting to maintain precision
df_results.to_csv(csv_filename, index=False, float_format='%.8e')

print("Calculation complete.")
print("Results saved to " + csv_filename)
print("First few rows of the results:")
# .to_string() provides better console formatting for pandas DataFrames
print(df_results.head().to_string()) 
print("\nLast few rows of the results:")
print(df_results.tail().to_string())

# Print some summary statistics of the calculated relative difference
print("\nSummary statistics for relative difference:")
print("Min rel_diff: " + str(np.min(relative_difference)))
print("Max rel_diff: " + str(np.max(relative_difference)))
print("Mean rel_diff: " + str(np.mean(relative_difference)))
