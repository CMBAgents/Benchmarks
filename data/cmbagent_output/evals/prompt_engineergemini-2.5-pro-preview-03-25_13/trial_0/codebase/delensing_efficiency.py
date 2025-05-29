# filename: codebase/delensing_efficiency.py
import os
import numpy as np
import pandas as pd
import camb
from camb.model import CAMBparams

# Ensure the output directory exists
os.makedirs("data", exist_ok=True)

# Cosmological parameters
H0 = 67.5  # Hubble constant in km/s/Mpc
ombh2 = 0.022  # Baryon density
omch2 = 0.122  # Cold dark matter density
mnu = 0.06  # Sum of neutrino masses in eV
omk = 0.0  # Curvature
tau = 0.06  # Optical depth to reionization
As = 2e-9  # Scalar amplitude
ns = 0.965  # Scalar spectral index

# Multipole limits
lmax_camb_internal = 2500  # Max l for CAMB calculations (Params.max_l)
lmax_working = 2000        # Max l for intermediate calculations involving N0

# --- Step 1: Set up CAMB parameters ---
pars = CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars.InitPower.set_params(As=As, ns=ns, r=0)  # r=0 for no tensors
pars.set_for_lmax(lmax_camb_internal, lens_potential_accuracy=1)  # Calculate C_ell^phiphi
pars.WantScalars = True
pars.WantTensors = False  # Not requesting tensor modes

# --- Step 2: First CAMB Run (Lensed Spectra) ---
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=False)  # raw_cl=False gives D_ells

# Lensed B-mode power spectrum (C_ell^BB)
# powers['lensed_scalar'] has columns TT, EE, BB, TE. Index 2 is BB.
# These are D_ell = ell(ell+1)C_ell/(2pi) in muK^2
dl_bb_lensed = powers['lensed_scalar'][:, 2]  # D_ell^BB from ell=0 to lmax_camb_internal
ells_camb = np.arange(lmax_camb_internal + 1)
cl_bb_lensed = np.zeros(lmax_camb_internal + 1)  # C_ell^BB in muK^2
# Conversion D_ell to C_ell: C_ell = D_ell * 2pi / (ell(ell+1))
# Avoid division by zero for ell=0,1. C_ell is 0 for ell=0,1.
non_zero_ells_mask = (ells_camb >= 2)
ells_valid = ells_camb[non_zero_ells_mask]
cl_bb_lensed[non_zero_ells_mask] = dl_bb_lensed[non_zero_ells_mask] * (2 * np.pi) / (ells_valid * (ells_valid + 1))

# CMB lensing potential power spectrum (C_ell^phiphi)
# results.get_lensing_potential_cls() returns C_ell^phiphi (dimensionless)
# The [:,0] selects C_ell^phiphi (other columns could be cross-spectra like C_ell^phiaT)
camb_cl_pp_raw = results.get_lensing_potential_cls(lmax=lmax_camb_internal)[:, 0]  # C_ell^phiphi from ell=0 to lmax_camb_internal

# --- Step 3: Load Lensing Noise Power Spectrum (N0) ---
noise_file_path = '/Users/antoidicherianlonappan/Workspace/Benchmarks/examples/../benchmarks/../data/N0.csv'
# The problem states Nl in file is N0 * (ell(ell+1))^2 / (2pi)
try:
    df_n0 = pd.read_csv(noise_file_path)
except Exception as e:
    print("Error loading N0.csv: " + str(e))
    raise

# --- Step 4: Process C_ell^phiphi and N0 up to lmax_working ---
# Take C_ell^phiphi up to lmax_working
cl_pp_raw_working = camb_cl_pp_raw[0:lmax_working + 1]

# Calculate C_ell^phiphi_scaled = C_ell^phiphi * (ell(ell+1))^2 / (2pi)
ls_working = np.arange(lmax_working + 1)
cl_pp_scaled_working = np.zeros(lmax_working + 1)
# factor_scaling is (ell(ell+1))^2 / (2pi)
factor_scaling_numerator = (ls_working * (ls_working + 1))**2
factor_scaling_denominator = (2 * np.pi)
mask_ls_ge2_working = (ls_working >= 2)
cl_pp_scaled_working[mask_ls_ge2_working] = cl_pp_raw_working[mask_ls_ge2_working] * factor_scaling_numerator[mask_ls_ge2_working] / factor_scaling_denominator

# Align N0 to ls_working
n0_scaled_from_file = np.full(lmax_working + 1, np.inf)  # Use np.inf for missing noise values

df_n0_filtered = df_n0[df_n0['l'] <= lmax_working]
valid_indices_l = df_n0_filtered['l'].values.astype(int)
valid_nl_values = df_n0_filtered['Nl'].values
mask_indices_in_range = valid_indices_l <= lmax_working
actual_indices = valid_indices_l[mask_indices_in_range]
actual_values = valid_nl_values[mask_indices_in_range]
safe_indices_mask = actual_indices < len(n0_scaled_from_file)
n0_scaled_from_file[actual_indices[safe_indices_mask]] = actual_values[safe_indices_mask]

# --- Step 5: Calculate Residual Lensing Potential Power Spectrum (scaled) ---
cl_pp_res_scaled = np.zeros(lmax_working + 1)
cl_pp_slice = cl_pp_scaled_working[mask_ls_ge2_working]
# n0_scaled_from_file also needs to be sliced for ell >= 2
n0_slice = n0_scaled_from_file[ls_working[mask_ls_ge2_working]]

denominator = cl_pp_slice + n0_slice
ratio = np.zeros_like(denominator)
valid_den_mask = (denominator > 0) & np.isfinite(denominator)
ratio[valid_den_mask] = cl_pp_slice[valid_den_mask] / denominator[valid_den_mask]
cl_pp_res_scaled[mask_ls_ge2_working] = cl_pp_slice * (1 - ratio)

# --- Step 6: Convert C_ell^phiphi_res_scaled to unscaled C_ell^phiphi_res and Pad ---
cl_pp_res_unscaled_working = np.zeros(lmax_working + 1)
# Instead of using advanced indexing on a copy, get the actual indices for ell >=2
indices_ge2 = np.where(mask_ls_ge2_working)[0]
# For these indices, compute the unscaled value: C_ell^phiphi_res_unscaled = C_ell^phiphi_res_scaled * (2pi) / (ell(ell+1))^2
# Avoid division by zero is already ensured since ell>=2
for idx in indices_ge2:
    if factor_scaling_numerator[idx] > 0:
        cl_pp_res_unscaled_working[idx] = cl_pp_res_scaled[idx] * factor_scaling_denominator / factor_scaling_numerator[idx]

# Pad to lmax_camb_internal (Params.max_l)
cl_pp_res_custom_for_camb = np.zeros(lmax_camb_internal + 1)
len_to_copy = min(lmax_working + 1, lmax_camb_internal + 1)
cl_pp_res_custom_for_camb[0:len_to_copy] = cl_pp_res_unscaled_working[0:len_to_copy]

# --- Step 7: Second CAMB Run (Delensed Spectra) ---
pars_delensed = CAMBparams()
pars_delensed.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
pars_delensed.InitPower.set_params(As=As, ns=ns, r=0)
pars_delensed.set_for_lmax(lmax_camb_internal, lens_potential_accuracy=0)  # lens_potential_accuracy=0 for custom
pars_delensed.WantScalars = True
pars_delensed.WantTensors = False

pars_delensed.set_custom_scalar_lensing_potential(cl_pp_res_custom_for_camb)

results_delensed = camb.get_results(pars_delensed)
powers_delensed = results_delensed.get_cmb_power_spectra(pars_delensed, CMB_unit='muK', raw_cl=False)

dl_bb_delensed = powers_delensed['lensed_scalar'][:, 2]
cl_bb_delensed = np.zeros(lmax_camb_internal + 1)
cl_bb_delensed[non_zero_ells_mask] = dl_bb_delensed[non_zero_ells_mask] * (2 * np.pi) / (ells_valid * (ells_valid + 1))

# --- Step 8: Calculate Delensing Efficiency ---
l_min_out = 2
l_max_out = 100
ells_out_indices = np.arange(l_min_out, l_max_out + 1)

cl_bb_lensed_out = cl_bb_lensed[l_min_out : l_max_out + 1]
cl_bb_delensed_out = cl_bb_delensed[l_min_out : l_max_out + 1]

efficiency_out = np.zeros_like(cl_bb_lensed_out)
valid_eff_mask = cl_bb_lensed_out != 0
efficiency_out[valid_eff_mask] = 100 * (cl_bb_lensed_out[valid_eff_mask] - cl_bb_delensed_out[valid_eff_mask]) / cl_bb_lensed_out[valid_eff_mask]
efficiency_out[~valid_eff_mask] = 0.0  # Handle cases where lensed C_ell is 0

# --- Step 9: Save Results and Print Information ---
output_csv_path = 'data/result.csv'
df_output = pd.DataFrame({'l': ells_out_indices, 'delensing_efficiency': efficiency_out})
df_output.to_csv(output_csv_path, index=False, float_format='%.4f')

print("--- Cosmological Parameters Used ---")
print("H0: " + str(H0) + " km/s/Mpc")
print("ombh2: " + str(ombh2))
print("omch2: " + str(omch2))
print("mnu: " + str(mnu) + " eV")
print("omk: " + str(omk))
print("tau: " + str(tau))
print("As: " + str(As))
print("ns: " + str(ns))
print("\n--- Calculated Power Spectra Examples (at selected ell values) ---")

ells_to_print = [10, 50, 100, 1000, 2000]
print("ell | C_ell^BB_lensed (muK^2) | C_ell^BB_delensed (muK^2) | C_ell^phiphi_scaled (dimless) | Nl_noise_scaled (dimless)")
print("----|-------------------------|---------------------------|-------------------------------|---------------------------")
for l_val in ells_to_print:
    cbb_l, cbb_d, cpp_s, nl_s = np.nan, np.nan, np.nan, np.nan  # Default to NaN
    if l_val <= lmax_camb_internal:
        if l_val < len(cl_bb_lensed): 
            cbb_l = cl_bb_lensed[l_val]
        if l_val < len(cl_bb_delensed): 
            cbb_d = cl_bb_delensed[l_val]
    if l_val <= lmax_working:
        if l_val < len(cl_pp_scaled_working): 
            cpp_s = cl_pp_scaled_working[l_val]
        if l_val < len(n0_scaled_from_file): 
            nl_s = n0_scaled_from_file[l_val]
    print(str(l_val).ljust(4) + " | " + "{:.4e}".format(cbb_l).ljust(23) + " | " + "{:.4e}".format(cbb_d).ljust(25) + " | " + "{:.4e}".format(cpp_s).ljust(29) + " | " + "{:.4e}".format(nl_s))

print("\n--- Delensing Efficiency Results (l=2 to 100) ---")
with pd.option_context('display.max_rows', 10, 'display.precision', 4):
    print(df_output)

print("\nDelensing efficiency results saved to: " + output_csv_path)

if 2000 < len(cl_bb_lensed):
    print("\nValue of C_l^BB (lensed) at l=2000: {:.4e} muK^2".format(cl_bb_lensed[2000]))
else:
    print("\nValue of C_l^BB (lensed) at l=2000: Not available (lmax_camb_internal < 2000 or array too short)")

if 2000 < len(cl_pp_scaled_working):
    print("Value of C_l^phiphi_scaled at l=2000: {:.4e}".format(cl_pp_scaled_working[2000]))
else:
    print("Value of C_l^phiphi_scaled at l=2000: Not available (lmax_working < 2000 or array too short)")
