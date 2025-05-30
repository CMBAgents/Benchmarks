# filename: codebase/camb_neutrino_powerspectrum.py
import camb
import numpy as np

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
    # kmax for CAMB's internal calculation range.
    # Actual output k range is determined by 'ks' in get_matter_power_spectrum.
    pars.set_matter_power(redshifts=[z_pk], kmax=k_max_h_mpc, nonlinear=False)
    return pars

# Setup CAMB for normal hierarchy
print("Setting up CAMB for normal neutrino hierarchy...")
pars_normal = setup_camb_params('normal', mnu_sum_val)

# Setup CAMB for inverted hierarchy
print("Setting up CAMB for inverted neutrino hierarchy...")
pars_inverted = setup_camb_params('inverted', mnu_sum_val)

# Validation checks for parameters
print("\n--- Parameter Validation Checks ---")

# Check H0
if not (pars_normal.H0 == H0_val and pars_inverted.H0 == H0_val):
    raise ValueError("H0 not set correctly. Normal: " + str(pars_normal.H0) + ", Inverted: " + str(pars_inverted.H0))
print("H0: OK (Value: " + str(H0_val) + " km/s/Mpc)")

# Check ombh2
if not (pars_normal.ombh2 == ombh2_val and pars_inverted.ombh2 == ombh2_val):
    raise ValueError("ombh2 not set correctly. Normal: " + str(pars_normal.ombh2) + ", Inverted: " + str(pars_inverted.ombh2))
print("ombh2: OK (Value: " + str(ombh2_val) + ")")

# Check omch2
if not (pars_normal.omch2 == omch2_val and pars_inverted.omch2 == omch2_val):
    raise ValueError("omch2 not set correctly. Normal: " + str(pars_normal.omch2) + ", Inverted: " + str(pars_inverted.omch2))
print("omch2: OK (Value: " + str(omch2_val) + ")")

# Check omk
if not (pars_normal.omk == omk_val and pars_inverted.omk == omk_val):
    raise ValueError("omk not set correctly. Normal: " + str(pars_normal.omk) + ", Inverted: " + str(pars_inverted.omk))
print("omk: OK (Value: " + str(omk_val) + ")")

# Check As
if not (pars_normal.InitPower.As == As_val and pars_inverted.InitPower.As == As_val):
    raise ValueError("As not set correctly. Normal: " + str(pars_normal.InitPower.As) + ", Inverted: " + str(pars_inverted.InitPower.As))
print("As: OK (Value: " + str(As_val) + ")")

# Check ns
if not (pars_normal.InitPower.ns == ns_val and pars_inverted.InitPower.ns == ns_val):
    raise ValueError("ns not set correctly. Normal: " + str(pars_normal.InitPower.ns) + ", Inverted: " + str(pars_inverted.InitPower.ns))
print("ns: OK (Value: " + str(ns_val) + ")")

# Check neutrino parameters
# M_nu = 93.14 eV * omnuh2
expected_omnuh2 = mnu_sum_val / 93.14
if not (abs(pars_normal.omnuh2 - expected_omnuh2) < 1e-6 and abs(pars_inverted.omnuh2 - expected_omnuh2) < 1e-6): # Increased precision for check
    raise ValueError("omnuh2 not set correctly based on mnu. Normal: " + str(pars_normal.omnuh2) + ", Inverted: " + str(pars_inverted.omnuh2) + ", Expected: " + str(expected_omnuh2))
print("omnuh2 (derived from mnu): OK (Normal: " + str(pars_normal.omnuh2) + ", Inverted: " + str(pars_inverted.omnuh2) + ")")

if not (pars_normal.num_nu_massive == 3 and pars_inverted.num_nu_massive == 3):
     raise ValueError("Number of massive neutrinos not set to 3. Normal: " + str(pars_normal.num_nu_massive) + ", Inverted: " + str(pars_inverted.num_nu_massive))
print("Number of massive neutrinos: OK (Normal: " + str(pars_normal.num_nu_massive) + ", Inverted: " + str(pars_inverted.num_nu_massive) + ")")

# Check input mnu sum
if not (abs(pars_normal.mnu - mnu_sum_val) < 1e-9 and abs(pars_inverted.mnu - mnu_sum_val) < 1e-9): # mnu should be exactly what was passed
    raise ValueError("Input mnu sum not stored correctly. Normal: " + str(pars_normal.mnu) + ", Inverted: " + str(pars_inverted.mnu))
print("Input mnu sum: OK (Value: " + str(mnu_sum_val) + " eV)")

print("Individual neutrino masses for Normal Hierarchy (eV): " + str(pars_normal.nu_masses))
print("Sum of individual normal hierarchy masses: " + str(np.sum(pars_normal.nu_masses)) + " eV")
print("Individual neutrino masses for Inverted Hierarchy (eV): " + str(pars_inverted.nu_masses))
print("Sum of individual inverted hierarchy masses: " + str(np.sum(pars_inverted.nu_masses)) + " eV")
# Note: Sum of pars.nu_masses might slightly differ from pars.mnu due to internal CAMB calculations for splittings.
# The critical check is that omnuh2 is correctly derived from the input mnu.

print("\n--- Computing Power Spectra ---")

# Get results for normal hierarchy
print("Running CAMB for normal hierarchy...")
results_normal = camb.get_results(pars_normal)
# P(k/h) in (Mpc/h)^3, k in h/Mpc
# kh_normal: array of k values (h/Mpc)
# z_normal: array of redshifts
# Pk_normal_all_z: array of P(k) values, shape (len(kh_normal), len(z_normal))
kh_normal, z_normal, Pk_normal_all_z = results_normal.get_matter_power_spectrum(
    ks=k_values_h_mpc,    # k values in h/Mpc
    nonlinear=False,      # Request linear power spectrum
    k_hunit=True,         # Input ks are in h/Mpc
    hubble_units=True     # Output P(k) in (Mpc/h)^3
)
Pk_normal = Pk_normal_all_z[:, 0] # Extract P(k) at z_pk (first and only redshift)

# Get results for inverted hierarchy
print("Running CAMB for inverted hierarchy...")
results_inverted = camb.get_results(pars_inverted)
kh_inverted, z_inverted, Pk_inverted_all_z = results_inverted.get_matter_power_spectrum(
    ks=k_values_h_mpc,
    nonlinear=False,
    k_hunit=True,
    hubble_units=True
)
Pk_inverted = Pk_inverted_all_z[:, 0] # Extract P(k) at z_pk

# Validation for k-values
if not np.allclose(kh_normal, k_values_h_mpc):
    raise ValueError("Output k-values (kh_normal) do not match input k_values_h_mpc.")
if not np.allclose(kh_inverted, k_values_h_mpc):
    raise ValueError("Output k-values (kh_inverted) do not match input k_values_h_mpc.")
print("k-values: OK (Output k matches input k for both hierarchies)")

# Check redshift
if not (len(z_normal) == 1 and abs(z_normal[0] - z_pk) < 1e-6):
    raise ValueError("Redshift for normal hierarchy P(k) is not " + str(z_pk) + ". Got: " + str(z_normal))
if not (len(z_inverted) == 1 and abs(z_inverted[0] - z_pk) < 1e-6):
    raise ValueError("Redshift for inverted hierarchy P(k) is not " + str(z_pk) + ". Got: " + str(z_inverted))
print("Redshift: OK (P(k) computed at z=" + str(z_pk) + " for both hierarchies)")


# Print shapes of P(k) arrays
print("Shape of P(k) for normal hierarchy: " + str(Pk_normal.shape)) # (n_k_points,)
print("Shape of P(k) for inverted hierarchy: " + str(Pk_inverted.shape)) # (n_k_points,)

# Print some P(k) values as a sample
print("\nSample P(k) values (first 5):")
print("k (h/Mpc) | P(k) Normal [(Mpc/h)^3] | P(k) Inverted [(Mpc/h)^3]")
for i in range(min(5, n_k_points)):
    print(str(k_values_h_mpc[i]) + " | " + str(Pk_normal[i]) + " | " + str(Pk_inverted[i]))

print("\nLinear matter power spectra P(k) computed successfully for both hierarchies.")
# Variables Pk_normal, Pk_inverted, and k_values_h_mpc are now available.
