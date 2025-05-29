# filename: codebase/camb_power_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

# Cosmological parameters
H0_param = 67.5  # Hubble constant in km/s/Mpc
ombh2_param = 0.022  # Baryon density Omega_b * h^2
omch2_param = 0.122  # Cold dark matter density Omega_c * h^2
mnu_param = 0.06  # Sum of neutrino masses in eV
omk_param = 0.0  # Curvature Omega_k
tau_param = 0.06  # Optical depth to reionization
r_tensor_to_scalar_param = 0.0  # Tensor-to-scalar ratio
As_scalar_amp_param = 2e-9  # Scalar amplitude A_s
ns_scalar_idx_param = 0.965  # Scalar spectral index n_s

l_max_calculation = 3050  # Max multipole for CAMB's internal calculation
l_max_output = 3000  # Max multipole for the output spectra

# Create 'data' directory if it doesn't exist
data_dir_path = "data"
os.makedirs(data_dir_path, exist_ok=True)

# Set up CAMB parameters
pars = camb.CAMBparams()
pars.set_cosmology(
    H0=H0_param,
    ombh2=ombh2_param,
    omch2=omch2_param,
    mnu=mnu_param,
    omk=omk_param,
    tau=tau_param
)
pars.InitPower.set_params(
    As=As_scalar_amp_param,
    ns=ns_scalar_idx_param,
    r=r_tensor_to_scalar_param
)

# Ensure lensing is on (it's on by default if AccuracyBoost >=1, but explicit is good)
pars.DoLensing = True

# Set lmax for calculation and accuracy for lensing potential
# lens_potential_accuracy: 0=fastest, 1=default, increase for more accuracy if needed
pars.set_for_lmax(lmax=l_max_calculation, lens_potential_accuracy=1)

# Calculate results (this can take a moment)
print("Running CAMB to calculate power spectra... This may take a few moments.")
results = camb.get_results(pars)
print("CAMB calculation complete.")

# Get CMB power spectra: D_l = l(l+1)C_l/(2*pi)
# spectra=['lensed_scalar'] is appropriate as r=0, so B-modes are from lensing.
# CMB_unit='muK' ensures D_l is in muK^2.
# raw_cl=False (default) gives D_l, not C_l.
power_spectra = results.get_cmb_power_spectra(
    lmax=l_max_output,
    spectra=['lensed_scalar'],
    CMB_unit='muK'
)

# power_spectra['lensed_scalar'] is an array of shape (l_max_output + 1, 4)
# Columns are TT, EE, BB, TE. We need BB (index 2).
# The array is indexed by l, from l=0 to l_max_output.
all_multipoles = np.arange(l_max_output + 1)  # ls from 0 to l_max_output
# D_l^BB values for all ls
dl_bb_all_values = power_spectra['lensed_scalar'][:, 2]

# Select multipoles from l=2 to l_max_output
# Indices for l=2 to l_max_output are 2 to l_max_output inclusive in the array
selected_multipoles = all_multipoles[2:(l_max_output + 1)] 
selected_dl_bb_values = dl_bb_all_values[2:(l_max_output + 1)]

# Create Pandas DataFrame
results_df = pd.DataFrame({
    'l': selected_multipoles.astype(int),
    'BB': selected_dl_bb_values
})

# Save to CSV file
output_csv_filename = os.path.join(data_dir_path, "result.csv")
results_df.to_csv(output_csv_filename, index=False)
print("B-mode polarization power spectrum saved to: " + output_csv_filename)

# Print summary information to console
print("\nSummary of the calculated B-mode power spectrum (l(l+1)C_l^BB/(2pi) in muK^2):")
if not results_df.empty:
    print("Number of l values: " + str(len(results_df)))
    print("Min l: " + str(results_df['l'].min()) + ", Max l: " + str(results_df['l'].max()))

    print("\nFirst 5 rows of the data:")
    print(results_df.head().to_string())

    print("\nLast 5 rows of the data:")
    print(results_df.tail().to_string())

    # Print specific values for quick check
    value_at_l2 = results_df[results_df['l'] == 2]['BB'].iloc[0]
    print("\nValue at l=2: " + str(value_at_l2) + " muK^2")

    if l_max_output >= 100 and 100 in results_df['l'].values:
        value_at_l100 = results_df[results_df['l'] == 100]['BB'].iloc[0]
        print("Value at l=100: " + str(value_at_l100) + " muK^2")
    elif l_max_output >=100:
        print("Value at l=100 not available in the selected range.")

    
    if l_max_output in results_df['l'].values:
        value_at_l_max = results_df[results_df['l'] == l_max_output]['BB'].iloc[0]
        print("Value at l=" + str(l_max_output) + ": " + str(value_at_l_max) + " muK^2")
    else:
        print("Value at l=" + str(l_max_output) + " not available in the selected range.")
else:
    print("No data to display. The DataFrame is empty.")
