# filename: codebase/camb_power_spectrum.py
import camb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

# Ensure the data directory exists
database_path = 'data/'
if not os.path.exists(database_path):
    os.makedirs(database_path)

# Disable LaTeX rendering for matplotlib
plt.rcParams['text.usetex'] = False

# Cosmological parameters
H0_val = 67.5  # Hubble constant in km/s/Mpc
ombh2_val = 0.022  # Baryon density Omega_b * h^2
omch2_val = 0.122  # Cold dark matter density Omega_c * h^2
mnu_sum_val = 0.11  # Sum of neutrino masses in eV
As_val = 2e-9  # Scalar amplitude
ns_val = 0.965  # Scalar spectral index
omk_val = 0.0  # Curvature density Omega_k (flat universe)

# k-values for power spectrum
k_min_h_mpc = 1e-4  # Minimum k in h/Mpc
k_max_h_mpc = 2.0    # Maximum k in h/Mpc
n_k_points = 200
# k_values are in h/Mpc
k_values_h_mpc = np.linspace(k_min_h_mpc, k_max_h_mpc, n_k_points)

# Redshift for power spectrum
z_pk = 0.0 # Redshift

def setup_camb_params(hierarchy_type, mnu_val):
    """
    Sets up CAMB parameters for a given neutrino hierarchy and total mass.

    Parameters:
    hierarchy_type (str): Neutrino hierarchy, 'normal' or 'inverted'.
    mnu_val (float): Total sum of neutrino masses in eV.

    Returns:
    camb.CAMBparams: Configured CAMB parameters object.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=H0_val,
        ombh2=ombh2_val,
        omch2=omch2_val,
        mnu=mnu_val,
        omk=omk_val,
        neutrino_hierarchy=hierarchy_type,
        num_massive_neutrinos=3 # Explicitly set 3 massive neutrinos for hierarchical models
    )
    pars.InitPower.set_params(As=As_val, ns=ns_val)
    pars.set_matter_power(redshifts=[z_pk], kmax=k_max_h_mpc, nonlinear=False)
    return pars

# Setup CAMB for normal hierarchy
print("Setting up CAMB for normal neutrino hierarchy...")
pars_normal = setup_camb_params('normal', mnu_sum_val)

# Setup CAMB for inverted hierarchy
print("Setting up CAMB for inverted neutrino hierarchy...")
pars_inverted = setup_camb_params('inverted', mnu_sum_val)

# Validation checks (simplified for brevity in this final step, full checks in previous step)
print("\n--- Parameter Validation Summary ---")
if not (pars_normal.H0 == H0_val and pars_inverted.H0 == H0_val and
        pars_normal.ombh2 == ombh2_val and pars_inverted.ombh2 == ombh2_val and
        pars_normal.omch2 == omch2_val and pars_inverted.omch2 == omch2_val and
        pars_normal.omk == omk_val and pars_inverted.omk == omk_val and
        pars_normal.InitPower.As == As_val and pars_inverted.InitPower.As == As_val and
        pars_normal.InitPower.ns == ns_val and pars_inverted.InitPower.ns == ns_val and
        abs(pars_normal.mnu - mnu_sum_val) < 1e-9 and abs(pars_inverted.mnu - mnu_sum_val) < 1e-9 and
        pars_normal.num_nu_massive == 3 and pars_inverted.num_nu_massive == 3):
    raise ValueError("Basic parameter consistency check failed.")
print("Basic cosmological parameters are consistent between models.")
print("Individual neutrino masses for Normal Hierarchy (eV): " + str(pars_normal.nu_masses))
print("Sum of individual normal hierarchy masses: " + str(np.sum(pars_normal.nu_masses)) + " eV")
print("Individual neutrino masses for Inverted Hierarchy (eV): " + str(pars_inverted.nu_masses))
print("Sum of individual inverted hierarchy masses: " + str(np.sum(pars_inverted.nu_masses)) + " eV")


print("\n--- Computing Power Spectra ---")

# Get results for normal hierarchy
print("Running CAMB for normal hierarchy...")
results_normal = camb.get_results(pars_normal)
kh_normal, z_normal, Pk_normal_all_z = results_normal.get_matter_power_spectrum(
    ks=k_values_h_mpc,
    nonlinear=False,
    k_hunit=True,
    hubble_units=True
)
Pk_normal = Pk_normal_all_z[:, 0] # P(k) in (Mpc/h)^3

# Get results for inverted hierarchy
print("Running CAMB for inverted hierarchy...")
results_inverted = camb.get_results(pars_inverted)
kh_inverted, z_inverted, Pk_inverted_all_z = results_inverted.get_matter_power_spectrum(
    ks=k_values_h_mpc,
    nonlinear=False,
    k_hunit=True,
    hubble_units=True
)
Pk_inverted = Pk_inverted_all_z[:, 0] # P(k) in (Mpc/h)^3

# Validate k-values and redshift (simplified)
if not (np.allclose(kh_normal, k_values_h_mpc) and np.allclose(kh_inverted, k_values_h_mpc) and
        len(z_normal) == 1 and abs(z_normal[0] - z_pk) < 1e-6 and
        len(z_inverted) == 1 and abs(z_inverted[0] - z_pk) < 1e-6):
    raise ValueError("Power spectrum output k or z validation failed.")
print("Power spectra P(k) computed successfully for both hierarchies at z=" + str(z_pk) + ".")

# Calculate relative difference
# Handle potential division by zero or NaN if Pk_normal is zero, though unlikely for P(k) > 0
# Add a small epsilon to Pk_normal if necessary, or check for zeros.
# For P(k), values should be positive.
if np.any(Pk_normal <= 0):
    print("Warning: Pk_normal contains zero or negative values. This might lead to issues in relative difference calculation.")

rel_diff = (Pk_inverted / Pk_normal) - 1.0

print("\n--- Saving Results ---")
# Save results to CSV
output_df = pd.DataFrame({
    'k': k_values_h_mpc,  # k in h/Mpc
    'rel_diff': rel_diff  # Relative difference (P_inv / P_norm - 1)
})
csv_filename = os.path.join(database_path, 'result.csv')
output_df.to_csv(csv_filename, index=False)
print("Results saved to " + str(csv_filename))
print("First 5 rows of the saved data:")
print(output_df.head().to_string())


print("\n--- Generating Plot ---")
# Plot the relative difference
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k_values_h_mpc, rel_diff, label='(P_inv / P_norm) - 1')
ax.set_xscale('log') # k is often plotted on a log scale
ax.set_xlabel('k (h/Mpc)')
ax.set_ylabel('Relative Difference ((P_inv / P_norm) - 1)')
ax.set_title('Relative Difference in Matter Power Spectrum (Inverted vs Normal Hierarchy)')
ax.grid(True, which="both", ls="-", alpha=0.5)
ax.legend()
plt.tight_layout()

# Save the plot
timestamp = time.strftime("%Y%m%d-%H%M%S")
plot_filename = os.path.join(database_path, 'relative_power_spectrum_difference_plot_1_' + str(timestamp) + '.png')
plt.savefig(plot_filename, dpi=300)
print("Plot saved as " + str(plot_filename))
print("Description: Plot shows the relative difference in the linear matter power spectrum between inverted and normal neutrino hierarchies as a function of wavenumber k (h/Mpc). The x-axis is on a logarithmic scale.")

print("\nAll tasks completed.")