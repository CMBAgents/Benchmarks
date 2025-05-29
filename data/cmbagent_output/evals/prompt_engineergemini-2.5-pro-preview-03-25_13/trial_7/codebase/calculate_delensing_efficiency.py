# filename: codebase/calculate_delensing_efficiency.py
import os
import numpy as np
import pandas as pd
import camb

def calculate_delensing_efficiency():
    r"""
    Calculates the delensing efficiency of the CMB B-mode polarization power spectrum.

    The function performs the following steps:
    1. Sets up cosmological parameters and initializes CAMB.
    2. Calculates the lensed B-mode power spectrum (C_ell^BB) and the CMB lensing potential
       power spectrum (C_ell^phiphi) using an initial CAMB run.
    3. Loads the lensing noise power spectrum (N_ell^phiphi) from a specified CSV file.
    4. Processes the noise spectrum and extends it as needed.
    5. Computes the residual lensing potential power spectrum (C_ell^phiphi,res).
    6. Performs a second CAMB run using C_ell^phiphi,res to calculate the delensed
       B-mode power spectrum (C_ell^BB,delensed).
    7. Calculates the delensing efficiency as 100 * (C_ell^BB,lensed - C_ell^BB,delensed) / C_ell^BB,lensed.
    8. Saves the delensing efficiency for multipoles l=2 to l=100 into a CSV file.
    """

    # Cosmological Parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Baryon density omega_b * h^2
    omch2 = 0.122  # Cold dark matter density omega_c * h^2
    mnu = 0.06  # Neutrino mass sum in eV
    omk = 0.0  # Curvature Omega_k
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index

    LMAX_OUTPUT = 2000  # Max ell for C_ell^BB output and N_0 usage
    # Max ell for C_ell^phiphi arrays for CAMB internal calculations.
    LMAX_CAMB_PHI = 4000 # Max ell for lensing potential C_ell^phiphi


    # 1. Initial CAMB run (Lensed spectra)
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=0) # r=0 for no tensors initially
    pars.WantScalars = True
    pars.WantTensors = False 
    pars.DoLensing = True 
    
    pars.set_for_lmax(lmax=LMAX_CAMB_PHI, lens_potential_accuracy=1)

    results_lensed = camb.get_results(pars)

    powers_lensed = results_lensed.get_lensed_scalar_cls(lmax=LMAX_OUTPUT) # In muK^2
    ell_arr_out = np.arange(LMAX_OUTPUT + 1)
    dl_bb_lensed = powers_lensed['BB']  # D_ell^BB = ell(ell+1)C_ell^BB/(2pi), units muK^2
    
    cl_bb_lensed = np.zeros_like(dl_bb_lensed) # C_ell^BB, units muK^2
    cl_bb_lensed[2:] = dl_bb_lensed[2:] * 2 * np.pi / (ell_arr_out[2:] * (ell_arr_out[2:] + 1))

    cl_pp_camb_full = results_lensed.get_lens_potential_cls(lmax=LMAX_CAMB_PHI)[:,0] # C_ell^phiphi

    # 2. Load and Process Lensing Noise N_ell^phiphi
    noise_file = '/Users/antoidicherianlonappan/Workspace/Benchmarks/examples/../benchmarks/../data/N0.csv'
    try:
        noise_data = pd.read_csv(noise_file)
    except FileNotFoundError:
        print("Error: Noise file not found at " + str(noise_file))
        return

    nl_pp_noise_extended = np.full(LMAX_CAMB_PHI + 1, np.inf) # N_ell^phiphi,noise
    nl_pp_noise_extended[0:2] = np.inf 

    l_csv = noise_data['l'].values.astype(int)
    nl_csv = noise_data['Nl'].values # This is N_ell^phiphi,noise

    mask_csv_indices = (l_csv >= 0) & (l_csv <= LMAX_OUTPUT)
    l_csv_filt = l_csv[mask_csv_indices]
    nl_csv_filt = nl_csv[mask_csv_indices]
    
    valid_l_indices = l_csv_filt[l_csv_filt < len(nl_pp_noise_extended)]
    valid_nl_values = nl_csv_filt[l_csv_filt < len(nl_pp_noise_extended)]
    nl_pp_noise_extended[valid_l_indices] = valid_nl_values


    # 3. Calculate Residual Lensing Potential C_ell^phiphi,res
    cl_pp_res_full = np.zeros(LMAX_CAMB_PHI + 1) # C_ell^phiphi,res

    for l_idx in range(2, LMAX_CAMB_PHI + 1): 
        cl_phi_val = cl_pp_camb_full[l_idx]
        nl_phi_val = nl_pp_noise_extended[l_idx]

        if cl_phi_val == 0:
            cl_pp_res_full[l_idx] = 0
        elif np.isinf(nl_phi_val): 
            cl_pp_res_full[l_idx] = cl_phi_val
        elif nl_phi_val == 0: 
            cl_pp_res_full[l_idx] = 0
        else: 
            cl_pp_res_full[l_idx] = cl_phi_val * nl_phi_val / (cl_phi_val + nl_phi_val)

    # 4. Second CAMB run (Delensed spectra)
    pars_delensed = camb.CAMBparams()
    pars_delensed.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars_delensed.InitPower.set_params(As=As, ns=ns, r=0)
    pars_delensed.WantScalars = True
    pars_delensed.WantTensors = False
    pars_delensed.DoLensing = True 

    pars_delensed.set_for_lmax(lmax=LMAX_CAMB_PHI, lens_potential_accuracy=1)
    
    pars_delensed.set_custom_scalar_cls(lmax=LMAX_CAMB_PHI, cl_phi=cl_pp_res_full)

    results_delensed = camb.get_results(pars_delensed)

    powers_delensed = results_delensed.get_lensed_scalar_cls(lmax=LMAX_OUTPUT) 
    dl_bb_delensed = powers_delensed['BB'] # D_ell^BB,delensed, units muK^2

    cl_bb_delensed = np.zeros_like(dl_bb_delensed) # C_ell^BB,delensed, units muK^2
    cl_bb_delensed[2:] = dl_bb_delensed[2:] * 2 * np.pi / (ell_arr_out[2:] * (ell_arr_out[2:] + 1))

    # 5. Calculate Delensing Efficiency
    delensing_efficiency = np.zeros(LMAX_OUTPUT + 1) # Percentage
    
    mask_eff = (cl_bb_lensed != 0) & (ell_arr_out >= 2)
    delensing_efficiency[mask_eff] = 100.0 * \
        (cl_bb_lensed[mask_eff] - cl_bb_delensed[mask_eff]) / cl_bb_lensed[mask_eff]
    
    mask_zero_lensed = (cl_bb_lensed == 0) & (ell_arr_out >= 2)
    delensing_efficiency[mask_zero_lensed] = 0.0


    # 6. Save Results
    l_min_out = 2
    l_max_out = 100
    
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    output_l_values = np.arange(l_min_out, l_max_out + 1)
    
    # Ensure output_l_values are valid indices for delensing_efficiency array
    # This step is important if l_max_out could exceed LMAX_OUTPUT
    valid_indices_for_output = output_l_values[output_l_values < len(delensing_efficiency)]
    output_efficiency_values = delensing_efficiency[valid_indices_for_output]

    output_df = pd.DataFrame({
        'l': valid_indices_for_output, 
        'delensing_efficiency': output_efficiency_values
    })
    
    output_filename = os.path.join(data_dir, 'result.csv')
    output_df.to_csv(output_filename, index=False)

    print("Delensing efficiency calculation complete.")
    print("Results saved to " + output_filename)
    
    print("\nSummary of Delensing Efficiency (%):")
    for l_val in [2, 10, 20, 50, 100]:
        if l_val <= l_max_out and l_val >= l_min_out :
            eff_series = output_df[output_df['l'] == l_val]['delensing_efficiency']
            if not eff_series.empty:
                print("l = " + str(l_val) + ": " + str(eff_series.iloc[0]))
            else:
                 print("l = " + str(l_val) + ": Not available in output range (check LMAX_OUTPUT vs l_max_out)")


if __name__ == '__main__':
    calculate_delensing_efficiency()