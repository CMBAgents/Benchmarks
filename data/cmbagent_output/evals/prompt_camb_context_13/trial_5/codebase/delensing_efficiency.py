# filename: codebase/delensing_efficiency.py
r"""
Calculate the delensing efficiency of the CMB B-mode polarization power spectrum
for a flat Lambda CDM cosmology using CAMB and an external lensing noise power spectrum.

Steps:
1. Set up cosmological parameters in CAMB.
2. Compute lensed B-mode and lensing potential power spectra up to l=2000.
3. Load and align external lensing noise power spectrum.
4. Calculate the residual lensing potential power spectrum.
5. Pad the residual spectrum to CAMB's required length.
6. Compute the delensed B-mode power spectrum.
7. Calculate delensing efficiency for l=2 to l=100.
8. Save results to 'data/result.csv' and print summary.

All units are annotated in the code. The output CSV contains columns:
- l: Multipole moment (integer, 2 to 100)
- delensing_efficiency: Delensing efficiency in percent

Author: Engineer Agent
"""

import camb
import numpy as np
import pandas as pd
import os

def main():
    r"""
    Main function to compute and save the delensing efficiency of the CMB B-mode polarization power spectrum.

    Saves the results to 'data/result.csv' and prints a detailed summary to the console.
    """
    # --- Cosmological Parameters (SI units where applicable) ---
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    As = 2.0e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]

    lmax_out = 2000  # Maximum multipole for output [dimensionless]
    lmax_calc = 3000  # Maximum multipole for CAMB internal calculation [dimensionless]
    lens_potential_accuracy_val = 2  # Lensing accuracy (higher is more accurate)

    # --- Setup CAMB ---
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax_calc, lens_potential_accuracy=lens_potential_accuracy_val)
    pars.WantTensors = False  # Only scalar-induced B-modes

    # --- Run CAMB ---
    results = camb.get_results(pars)

    # --- 1. Lensed B-mode power spectrum (D_ell^BB) [muK^2] ---
    lensed_cls = results.get_lensed_scalar_cls(CMB_unit='muK', lmax=lmax_out)
    dl_bb_lensed = lensed_cls[:, 2]  # D_ell^BB, [muK^2], index 2 is BB

    # --- 2. CMB lensing potential power spectrum ([L(L+1)]^2 C_L^phiphi / (2pi)) [dimensionless] ---
    cl_pp_camb_output = results.get_lens_potential_cls(lmax=lmax_out)
    cl_pp = cl_pp_camb_output[:, 0]  # [L(L+1)]^2 C_L^phiphi / (2pi), index 0 is PP

    # --- 3. Load N0 and align arrays ---
    n0_file_path = '/Users/antoidicherianlonappan/Software/Benchmarks/data/N0.csv'
    data_n0 = pd.read_csv(n0_file_path)
    n0_array = np.full(lmax_out + 1, np.inf)
    file_ls = data_n0['l'].values.astype(int)
    file_Nls = data_n0['Nl'].values
    mask = (file_ls <= lmax_out) & (file_ls >= 0)
    valid_file_ls = file_ls[mask]
    valid_file_Nls = file_Nls[mask]
    n0_array[valid_file_ls] = valid_file_Nls

    # --- 4. Residual lensing potential power spectrum ---
    f_filter = np.zeros_like(cl_pp)
    valid_calc_mask = ((cl_pp + n0_array) > 0) & np.isfinite(n0_array)
    f_filter[valid_calc_mask] = cl_pp[valid_calc_mask] / (cl_pp[valid_calc_mask] + n0_array[valid_calc_mask])
    perfect_recon_mask = (n0_array == 0) & (cl_pp > 0)
    f_filter[perfect_recon_mask] = 1.0
    cl_pp_res = cl_pp * (1 - f_filter)
    cl_pp_res[0:2] = 0.0  # Set l=0,1 to zero

    # --- 5. Pad the residual lensing potential array to CAMB's max_l ---
    cl_pp_res_padded = np.zeros(pars.max_l + 1)
    len_to_copy = min(len(cl_pp_res), len(cl_pp_res_padded))
    cl_pp_res_padded[:len_to_copy] = cl_pp_res[:len_to_copy]

    # --- 6. Delensed B-mode power spectrum (D_ell^BB_delensed) [muK^2] ---
    delensed_cls_output = results.get_lensed_cls_with_spectrum(
        clpp=cl_pp_res_padded, CMB_unit='muK', lmax=lmax_out
    )
    dl_bb_delensed = delensed_cls_output[:, 2]  # D_ell^BB_delensed, [muK^2]

    # --- 7. Delensing efficiency for l=2 to l=100 ---
    ls_for_eff = np.arange(2, 101)  # Multipoles for efficiency calculation
    efficiency = np.zeros_like(ls_for_eff, dtype=float)
    mask_lensed_gt_zero = dl_bb_lensed[ls_for_eff] > 1e-30
    efficiency[mask_lensed_gt_zero] = 100.0 * (
        dl_bb_lensed[ls_for_eff][mask_lensed_gt_zero] - dl_bb_delensed[ls_for_eff][mask_lensed_gt_zero]
    ) / dl_bb_lensed[ls_for_eff][mask_lensed_gt_zero]

    # --- 8. Save results to CSV ---
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "result.csv")
    output_df = pd.DataFrame({'l': ls_for_eff, 'delensing_efficiency': efficiency})
    output_df.to_csv(output_path, index=False)

    # --- Print detailed summary to console ---
    pd.set_option('display.float_format', lambda x: '%.6e' % x)
    print("\nDelensing efficiency calculation complete.")
    print("Cosmological parameters used:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  ombh2 = " + str(ombh2))
    print("  omch2 = " + str(omch2))
    print("  mnu = " + str(mnu) + " eV")
    print("  omk = " + str(omk))
    print("  tau = " + str(tau))
    print("  As = " + str(As))
    print("  ns = " + str(ns))
    print("Lensed and delensed B-mode power spectra computed up to l = " + str(lmax_out))
    print("Lensing noise loaded from: " + n0_file_path)
    print("Results saved to: " + output_path)
    print("\nDelensing efficiency (percent) for l=2 to l=100:\n")
    print(output_df.to_string(index=False))


if __name__ == "__main__":
    main()