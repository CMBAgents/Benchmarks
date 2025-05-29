# filename: codebase/delensing_efficiency.py
import os
import numpy as np
import pandas as pd
import camb
from camb.sources import GaussianSourceWindow, SplinedSourceWindow

def calculate_delensing_efficiency():
    r"""
    Calculates the delensing efficiency of the CMB B-mode polarization power spectrum.

    The calculation follows these steps:
    1. Set up CAMB with specified cosmological parameters.
    2. Compute the lensed B-mode power spectrum (C_ell^BB) and the CMB lensing potential power spectrum (C_ell^phiphi).
    3. Load the lensing noise power spectrum (N_ell^phiphi) from a CSV file.
    4. Align the lensing potential and noise power spectra arrays up to l=2000.
    5. Calculate the residual lensing potential power spectrum: C_ell^phiphi_res = C_ell^phiphi * (1 - (C_ell^phiphi / (C_ell^phiphi + N_ell^phiphi))).
    6. Pad the residual lensing potential array to CAMB's max_l.
    7. Compute the delensed B-mode power spectrum using the residual lensing potential.
    8. Calculate the delensing efficiency: 100 * (C_ell^BB_lensed - C_ell^BB_delensed) / C_ell^BB_lensed.
    9. Save the delensing efficiency for l=2 to l=100 to a CSV file.
    """

    # Cosmological parameters
    h0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Baryon density
    omch2 = 0.122  # Cold dark matter density
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    
    camb_lmax = 2500 # Max multipole for CAMB calculations
    analysis_lmax = 2000 # Max multipole for using N0 data

    # Setup CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(camb_lmax, lens_potential_accuracy=1) 
    pars.WantTensors = True 
    pars.DoLensing = True

    # Get results from CAMB
    results = camb.get_results(pars)
    # powers are D_ell = ell(ell+1)C_ell/(2pi) for TT,EE,BB,TE
    # lens_potential is C_ell^phiphi
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=False)

    # 1. Lensed B-mode power spectrum (C_ell^BB) in muK^2
    ells_camb = np.arange(powers['lensed_scalar'].shape[0])
    cl_conversion_factor = np.zeros_like(ells_camb, dtype=float)
    # C_ell = D_ell * 2pi / (ell(ell+1))
    non_zero_ells_mask = ells_camb > 1
    cl_conversion_factor[non_zero_ells_mask] = 2 * np.pi / (ells_camb[non_zero_ells_mask] * (ells_camb[non_zero_ells_mask] + 1))
    
    cl_lensed_bb = powers['lensed_scalar'][:,2] * cl_conversion_factor  # BB is index 2
    cl_lensed_bb[0:2] = 0.0  # Ensure C_l for l=0,1 are zero

    print("Lensed B-mode power spectrum C_ell^BB (muK^2):")
    print_sample_cls(cl_lensed_bb, ells_camb, l_values=[10, 100, 1000, min(2000, camb_lmax)])


    # 2. CMB lensing potential power spectrum (C_ell^phiphi * (ell(ell+1))^2 / (2pi))
    camb_cl_phi_phi = powers['lens_potential'][:,0]  # This is C_L^phiphi
    if len(camb_cl_phi_phi) < camb_lmax + 1:  # Pad if shorter than camb_lmax
        camb_cl_phi_phi = np.pad(camb_cl_phi_phi, (0, camb_lmax + 1 - len(camb_cl_phi_phi)), 'constant')
    else:  # Truncate if longer (should not happen with set_for_lmax)
        camb_cl_phi_phi = camb_cl_phi_phi[:camb_lmax+1]


    ells_for_kappa_like = np.arange(len(camb_cl_phi_phi))
    # Requested quantity: C_ell^phiphi * (ell(ell+1))^2 / (2pi)
    cl_kappa_like = camb_cl_phi_phi * (ells_for_kappa_like * (ells_for_kappa_like + 1))**2 / (2 * np.pi)
    cl_kappa_like[0:2] = 0.0 

    print("\nCMB lensing potential related spectrum C_ell^phiphi * (ell(ell+1))^2 / (2pi):")
    print_sample_cls(cl_kappa_like, ells_for_kappa_like, l_values=[10, 100, 1000, min(2000, camb_lmax)])


    # 3. Load lensing noise power spectrum (N_ell^phiphi)
    n0_file_path = '/Users/antoidicherianlonappan/Workspace/Benchmarks/data/N0.csv'
    try:
        n0_df = pd.read_csv(n0_file_path)
    except Exception as e:
        print("Error loading N0.csv: " + str(e))
        return

    if n0_df.empty:
        print("N0.csv is empty. Assuming infinite noise (no delensing).")
        l_n0_file = np.array([])
        nl_n0_file = np.array([])
    else:
        l_n0_file = n0_df['l'].values
        nl_n0_file = n0_df['Nl'].values  # Assumed to be N_ell^phiphi

    n_phi_phi_eff = np.full(camb_lmax + 1, np.inf) 
    
    if len(l_n0_file) > 0:
        l_min_n0_file = int(np.ceil(l_n0_file.min()))
        l_max_n0_file = int(np.floor(l_n0_file.max()))
        
        l_start_delens = max(2, l_min_n0_file)
        l_end_delens = min(analysis_lmax, l_max_n0_file)

        if l_start_delens <= l_end_delens:
            ells_to_interp = np.arange(l_start_delens, l_end_delens + 1)
            interp_nl_n0_values = np.interp(ells_to_interp, l_n0_file, nl_n0_file)
            n_phi_phi_eff[ells_to_interp] = interp_nl_n0_values
        else:
            print("Warning: No valid range for N0 interpolation. Check N0.csv data and analysis_lmax.")
    else:
        print("Warning: N0.csv contains no data. Assuming infinite noise (no delensing).")


    # 4. Residual lensing potential power spectrum C_ell^phiphi_res
    cl_pp_res_full = np.copy(camb_cl_phi_phi) 

    # Define the range where N_phi_phi_eff is finite for calculation
    finite_noise_mask = np.isfinite(n_phi_phi_eff)
    # Also ensure l >= 2 for physical meaning
    valid_delens_mask = finite_noise_mask & (np.arange(camb_lmax + 1) >= 2)

    cl_pp_subset = camb_cl_phi_phi[valid_delens_mask]
    n_phi_phi_subset = n_phi_phi_eff[valid_delens_mask]
    
    suppression_factor = np.zeros_like(cl_pp_subset)
    denominator_supp = cl_pp_subset + n_phi_phi_subset
    
    # Calculate suppression where denominator is not zero
    calc_mask = (denominator_supp != 0)
    suppression_factor[calc_mask] = cl_pp_subset[calc_mask] / denominator_supp[calc_mask]
    
    # Handle perfect reconstruction (N_phi_phi = 0 implies suppression = 1 if C_phi_phi != 0)
    perfect_recon_mask = (n_phi_phi_subset == 0) & (cl_pp_subset != 0)
    suppression_factor[perfect_recon_mask] = 1.0
    
    cl_pp_res_full[valid_delens_mask] = cl_pp_subset * (1 - suppression_factor)
    cl_pp_res_full[0:2] = 0.0


    # 6. Delensed B-mode power spectrum C_ell^BB,delensed
    # get_delensed_scalar_cls returns D_L = L(L+1)C_L/2pi
    delensed_powers_Dl = results.get_delensed_scalar_cls(cl_pp_res_full) 
    
    # Ensure cl_conversion_factor matches length of delensed_powers_Dl if different from ells_camb
    if len(cl_conversion_factor) != delensed_powers_Dl.shape[0]:
        temp_ells = np.arange(delensed_powers_Dl.shape[0])
        cl_conversion_factor_delensed = np.zeros_like(temp_ells, dtype=float)
        non_zero_mask_delensed = temp_ells > 1
        cl_conversion_factor_delensed[non_zero_mask_delensed] = 2 * np.pi / (temp_ells[non_zero_mask_delensed] * (temp_ells[non_zero_mask_delensed] + 1))
    else:
        cl_conversion_factor_delensed = cl_conversion_factor

    cl_bb_delensed = delensed_powers_Dl[:,2] * cl_conversion_factor_delensed 
    cl_bb_delensed[0:2] = 0.0


    # 7. Delensing efficiency
    delensing_efficiency = np.zeros_like(cl_lensed_bb)
    valid_l_indices_eff = cl_lensed_bb != 0
    
    # Ensure cl_bb_delensed is same length as cl_lensed_bb for direct operations
    # This should be guaranteed by CAMB using same lmax.
    min_len = min(len(cl_lensed_bb), len(cl_bb_delensed))
    cl_lensed_bb_eff = cl_lensed_bb[:min_len]
    cl_bb_delensed_eff = cl_bb_delensed[:min_len]
    valid_l_indices_eff = valid_l_indices_eff[:min_len]

    numerator_eff = cl_lensed_bb_eff[valid_l_indices_eff] - cl_bb_delensed_eff[valid_l_indices_eff]
    denominator_eff_val = cl_lensed_bb_eff[valid_l_indices_eff]
    
    delensing_efficiency_calc = np.zeros_like(cl_lensed_bb_eff)
    delensing_efficiency_calc[valid_l_indices_eff] = 100 * numerator_eff / denominator_eff_val
    
    # Final efficiency array, ensure it's full length matching ells_camb
    delensing_efficiency = np.zeros(len(ells_camb), dtype=float)
    delensing_efficiency[:min_len] = delensing_efficiency_calc


    # Save results to CSV
    output_l_min = 2
    output_l_max = 100
    output_ells = np.arange(output_l_min, output_l_max + 1)
    
    output_ells_valid = output_ells[output_ells < len(delensing_efficiency)]

    if len(output_ells_valid) > 0:
        df_results = pd.DataFrame({
            'l': output_ells_valid,
            'delensing_efficiency': delensing_efficiency[output_ells_valid]
        })

        os.makedirs("data", exist_ok=True)
        output_csv_path = "data/result.csv"
        df_results.to_csv(output_csv_path, index=False)
        print("\nDelensing efficiency results saved to: " + output_csv_path)
        print("Showing first few rows of the output data:")
        print(df_results.head().to_string())
    else:
        print("\nNo data to save for the specified l range [" + str(output_l_min) + ", " + str(output_l_max) + "].")

    print("\nSample values of C_ell^BB_lensed (muK^2):")
    print_sample_cls(cl_lensed_bb, ells_camb)
    
    print("\nSample values of C_ell^BB_delensed (muK^2):")
    print_sample_cls(cl_bb_delensed, ells_camb)  # Use ells_camb, assuming lengths match

    print("\nSample values of Delensing Efficiency (%):")
    print_sample_cls(delensing_efficiency, ells_camb, l_values=[2, 10, 50, 100])


def print_sample_cls(cls_array, ells_array, l_values=None):
    r"""Helper function to print sample Cl values."""
    if l_values is None:
        l_values = [10, 50, 100, 200, 500, 1000, 2000]
    
    print_ells = np.array(l_values)
    # Ensure l_values are within the bounds of the ells_array and cls_array
    print_ells = print_ells[print_ells < len(cls_array)]
    print_ells = print_ells[print_ells < len(ells_array)]

    for l_val in print_ells:
        # Assuming ells_array is 0, 1, 2,... so l_val is the index
        idx = l_val 
        print("  l=" + str(l_val) + ": " + str(cls_array[idx]))


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')  # Set non-interactive backend
    matplotlib.rcParams['text.usetex'] = False  # Disable LaTeX rendering
    
    calculate_delensing_efficiency()