# filename: codebase/camb_neutrino_power_spectrum.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # For log scale formatting (though not explicitly used for formatter object)
import camb
from camb import model
from scipy.optimize import root_scalar
import datetime

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# Neutrino mass splittings (eV^2)
# delta_m21_sq = m2^2 - m1^2
delta_m21_sq = 7.39e-5  # eV^2 (solar mass splitting)
# For Normal Hierarchy (NH): m1 < m2 < m3
# delta_m31_sq_NH = m3^2 - m1^2 (atmospheric mass splitting for NH)
delta_m31_sq_NH = 2.528e-3  # eV^2
# For Inverted Hierarchy (IH): m3 < m1 < m2
# abs_delta_m32_sq_IH is used in the solver for m1^2 - m3_lightest^2.
abs_delta_m32_sq_IH = 2.510e-3  # eV^2 (atmospheric mass splitting parameter for IH)

# Total neutrino mass sum (eV)
sum_mnu_eV = 0.11 # eV

def get_neutrino_masses(sum_mnu, hierarchy):
    """
    Calculates individual neutrino masses for a given hierarchy and sum of masses.

    Args:
        sum_mnu (float): Total sum of neutrino masses in eV.
        hierarchy (str): Neutrino hierarchy, 'normal' or 'inverted'.

    Returns:
        list: A sorted list of three neutrino masses [lightest, middle, heaviest] in eV.
    """
    if not (isinstance(sum_mnu, (float, int)) and sum_mnu > 0):
        raise ValueError("sum_mnu must be a positive number.")
    if hierarchy not in ['normal', 'inverted']:
        raise ValueError("hierarchy must be 'normal' or 'inverted'.")

    if hierarchy == 'normal':
        # Normal Hierarchy: m1_actual < m2_actual < m3_actual
        # Solves for m1_actual (lightest).
        # m2_actual^2 = m1_actual^2 + delta_m21_sq
        # m3_actual^2 = m1_actual^2 + delta_m31_sq_NH
        # Equation: m1_actual + sqrt(m1_actual^2 + delta_m21_sq) + sqrt(m1_actual^2 + delta_m31_sq_NH) - sum_mnu = 0
        def eqn_nh(m1_actual):
            if m1_actual < 0:
                return sum_mnu * 100
            m2_sq_val = m1_actual**2 + delta_m21_sq
            m3_sq_val = m1_actual**2 + delta_m31_sq_NH
            if m2_sq_val < 0 or m3_sq_val < 0:
                return sum_mnu * 100
            return m1_actual + np.sqrt(m2_sq_val) + np.sqrt(m3_sq_val) - sum_mnu
        
        try:
            sol = root_scalar(eqn_nh, bracket=[0, sum_mnu], x0=sum_mnu / 3.0, method='brentq')
            if not sol.converged:
                raise RuntimeError("Neutrino mass solver did not converge for normal hierarchy.")
            m1_actual = sol.root
        except ValueError as e:
            raise ValueError("Root finding failed for normal hierarchy. Check inputs or bracket: " + str(e))

        m2_actual = np.sqrt(m1_actual**2 + delta_m21_sq)
        m3_actual = np.sqrt(m1_actual**2 + delta_m31_sq_NH)
        return sorted([m1_actual, m2_actual, m3_actual])

    elif hierarchy == 'inverted':
        # Inverted Hierarchy: m3_actual < m1_actual < m2_actual
        # Solves for m3_actual (lightest).
        # m1_actual^2 = m3_actual^2 + abs_delta_m32_sq_IH 
        # m2_actual^2 = m1_actual^2 + delta_m21_sq = m3_actual^2 + abs_delta_m32_sq_IH + delta_m21_sq
        # Equation: m3_actual + sqrt(m3_actual^2 + abs_delta_m32_sq_IH) + sqrt(m3_actual^2 + abs_delta_m32_sq_IH + delta_m21_sq) - sum_mnu = 0
        def eqn_ih(m3_actual):
            if m3_actual < 0:
                return sum_mnu * 100
            m1_sq_val = m3_actual**2 + abs_delta_m32_sq_IH
            m2_sq_val = m3_actual**2 + abs_delta_m32_sq_IH + delta_m21_sq
            if m1_sq_val < 0 or m2_sq_val < 0:
                return sum_mnu * 100
            return m3_actual + np.sqrt(m1_sq_val) + np.sqrt(m2_sq_val) - sum_mnu

        try:
            sol = root_scalar(eqn_ih, bracket=[0, sum_mnu], x0=sum_mnu / 3.0, method='brentq')
            if not sol.converged:
                raise RuntimeError("Neutrino mass solver did not converge for inverted hierarchy.")
            m3_actual = sol.root
        except ValueError as e:
            raise ValueError("Root finding failed for inverted hierarchy. Check inputs or bracket: " + str(e))
            
        m1_actual = np.sqrt(m3_actual**2 + abs_delta_m32_sq_IH)
        m2_actual = np.sqrt(m3_actual**2 + abs_delta_m32_sq_IH + delta_m21_sq)
        return sorted([m3_actual, m1_actual, m2_actual])

# Cosmological parameters
H0 = 67.5  # Hubble constant in km/s/Mpc
ombh2 = 0.022  # Baryon density * h^2
omch2 = 0.122  # Cold dark matter density * h^2
As = 2e-9  # Scalar amplitude
ns = 0.965  # Scalar spectral index
h_param = H0 / 100.0  # Dimensionless Hubble parameter

# k-range for final output (linearly spaced)
k_min_h_Mpc = 1e-4  # Min k in h/Mpc
k_max_h_Mpc = 2.0   # Max k in h/Mpc
n_k_target = 200    # Number of target k points (linearly spaced)
k_target_array = np.linspace(k_min_h_Mpc, k_max_h_Mpc, n_k_target)  # Array of k in h/Mpc

# Parameters for CAMB's internal P(k) calculation (more points for smoother interpolation)
n_k_camb = 1000  # Number of log-spaced points for CAMB's internal calculation

def get_power_spectrum(hierarchy_type, target_k_values_h_mpc):
    """
    Computes the linear matter power spectrum for a given neutrino hierarchy,
    interpolated to target_k_values_h_mpc.

    Args:
        hierarchy_type (str): 'normal' or 'inverted'.
        target_k_values_h_mpc (np.ndarray): Array of k values (in h/Mpc) at which to interpolate P(k).

    Returns:
        np.ndarray: Interpolated P(k) values in (Mpc/h)^3 corresponding to target_k_values_h_mpc.
    """
    # Calculate individual neutrino masses
    mnu_list_eV = get_neutrino_masses(sum_mnu_eV, hierarchy_type)  # masses in eV
    
    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, omk=0.0, TCMB=2.7255)
    pars.InitPower.set_params(As=As, ns=ns, pivot_scalar=0.05)  # pivot_scalar is CAMB default
    
    pars.set_massivenu_params(mnu_list_eV[0], mnu_list_eV[1], mnu_list_eV[2], 
                              num_massive_neutrinos=3)

    pars.NonLinear = model.NonLinear_none  # Request linear power spectrum
    
    # kmax for set_matter_power is physical k (Mpc^-1).
    kmax_physical_limit_mpc_inv = (k_max_h_Mpc / h_param) * 1.05  # 5% safety margin
    pars.set_matter_power(redshifts=[0.], kmax=kmax_physical_limit_mpc_inv, 
                          accurate_massive_neutrino_transfers=True)
    
    try:
        results = camb.get_results(pars)
    except Exception as e:
        print("Error getting CAMB results for hierarchy: " + hierarchy_type)
        raise e

    # Get P(k) from CAMB on its internal (log-spaced) k grid
    # minkh, maxkh are in h/Mpc
    # P(k) units: (Mpc/h)^3
    kh_camb_h_mpc, _, pk_camb_raw_z_k = results.get_matter_power_spectrum(
        minkh=k_min_h_Mpc, maxkh=k_max_h_Mpc, npoints=n_k_camb,
        var1='delta_tot', var2='delta_tot',
        hubble_units=True, k_hunit=True, nonlinear=False
    )
    
    pk_camb_at_z0_mpch_cubed = pk_camb_raw_z_k[0]  # P(k) at z=0

    # Interpolate onto the target linearly spaced k grid
    if not (np.min(target_k_values_h_mpc) >= np.min(kh_camb_h_mpc) - 1e-9 and
            np.max(target_k_values_h_mpc) <= np.max(kh_camb_h_mpc) + 1e-9):
        print("Warning: Target k-values may extend slightly beyond CAMB's computed k-range due to precision.")
        print("Target k range (h/Mpc): " + str(np.min(target_k_values_h_mpc)) + " to " + str(np.max(target_k_values_h_mpc)))
        print("CAMB k range (h/Mpc):   " + str(np.min(kh_camb_h_mpc)) + " to " + str(np.max(kh_camb_h_mpc)))
        # np.interp will use boundary values for extrapolation if points are outside.
        # Given how ranges are set up, this should ideally not be a major issue.

    pk_interpolated_mpch_cubed = np.interp(target_k_values_h_mpc, kh_camb_h_mpc, pk_camb_at_z0_mpch_cubed)
    
    return pk_interpolated_mpch_cubed

# Compute P(k) for normal hierarchy
print("Calculating P(k) for normal hierarchy...")
pk_normal_mpch_cubed = get_power_spectrum('normal', k_target_array)  # P(k) in (Mpc/h)^3

# Compute P(k) for inverted hierarchy
print("Calculating P(k) for inverted hierarchy...")
pk_inverted_mpch_cubed = get_power_spectrum('inverted', k_target_array)  # P(k) in (Mpc/h)^3

# Validate power spectra
if np.any(np.isnan(pk_normal_mpch_cubed)) or np.any(np.isnan(pk_inverted_mpch_cubed)):
    raise ValueError("NaNs found in computed power spectra. Check CAMB setup or interpolation.")
if np.any(pk_normal_mpch_cubed <= 0):
    print("Warning: Non-positive values found in normal hierarchy P(k). Relative difference might be ill-defined at those points.")

# Calculate relative difference: (P_inv / P_norm - 1)
epsilon = 1e-30  # Small number to prevent division by zero if P_norm is exactly zero
rel_diff = (pk_inverted_mpch_cubed / (pk_normal_mpch_cubed + epsilon)) - 1

# Save results to CSV
results_df = pd.DataFrame({
    'k': k_target_array,  # k in h/Mpc
    'rel_diff': rel_diff  # Relative difference (dimensionless)
})
csv_filename = "data/result.csv"
results_df.to_csv(csv_filename, index=False, float_format='%.8e')
print("Results saved to " + csv_filename)

# Compute summary statistics for the relative difference
min_rel_diff = np.min(rel_diff)
max_rel_diff = np.max(rel_diff)
mean_rel_diff = np.mean(rel_diff)

print("\nSummary Statistics for Relative Difference ((P_inv / P_norm) - 1):")
print("Minimum Relative Difference: " + str(min_rel_diff))
print("Maximum Relative Difference: " + str(max_rel_diff))
print("Mean Relative Difference:    " + str(mean_rel_diff))

# Plot the relative difference
plt.figure(figsize=(10, 6))
plt.plot(k_target_array, rel_diff, label='(P_inv / P_norm) - 1')
plt.xscale('log')  # Use log scale for k-axis for better visualization

plt.xlabel('Wavenumber k (h/Mpc)')
plt.ylabel('Relative Difference in P(k)')
plt.title('Relative Difference in P(k) (Inverted vs Normal Hierarchy)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.tight_layout()

# Save the plot
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
plot_filename = "data/relative_difference_pk_" + timestamp + ".png"
plt.savefig(plot_filename, dpi=300)
print("Plot saved to " + plot_filename)
print("Description of the plot: The plot shows the relative difference in the linear matter power spectrum P(k) between the inverted and normal neutrino hierarchy models, as a function of wavenumber k (h/Mpc). The k-axis is on a logarithmic scale, while the y-axis shows the dimensionless relative difference (P_inverted / P_normal - 1).")

plt.close()  # Explicitly close plot

print("\nScript finished successfully.")