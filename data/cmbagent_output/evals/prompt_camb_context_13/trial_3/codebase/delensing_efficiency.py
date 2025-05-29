# filename: codebase/delensing_efficiency.py
import camb
import numpy as np
import pandas as pd
import os

def main():
    r"""
    Calculate the delensing efficiency of the CMB B-mode polarization power spectrum for a flat Lambda CDM cosmology
    using CAMB and an external lensing noise power spectrum.

    Steps:
    1. Set up cosmological parameters in CAMB.
    2. Compute the lensed B-mode power spectrum and lensing potential power spectrum up to l=2000.
    3. Load the external lensing noise N0 from a CSV file and align it with the CAMB output.
    4. Compute the residual lensing potential power spectrum.
    5. Pad the residual spectrum to the length required by CAMB.
    6. Compute the delensed B-mode power spectrum using the residual lensing potential.
    7. Calculate the delensing efficiency for 2 <= l <= 100.
    8. Save the results to data/result.csv and print a detailed summary.

    Units:
    - All CMB power spectra are in micro-Kelvin squared (uK^2).
    - Lensing potential spectra are in CAMB's convention: [l(l+1)]^2 C_l^{phiphi} / (2pi).
    - Multipole l is dimensionless.
    - Delensing efficiency is in percent (%).
    """

    # --- Parameters ---
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    As = 2.0e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]

    lmax_out = 2000  # Maximum multipole for output
    lmax_camb = 3000  # Internal CAMB lmax for accuracy
    lens_potential_accuracy = 2  # Lensing accuracy (higher = more accurate, slower)

    # --- Setup CAMB ---
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax_camb, lens_potential_accuracy=lens_potential_accuracy)
    pars.WantTensors = False  # Only scalar-induced B-modes

    # --- Run CAMB ---
    results = camb.get_results(pars)

    # --- 1. Lensed B-mode power spectrum (D_l^BB) [uK^2] ---
    lensed_cls = results.get_lensed_scalar_cls(CMB_unit='muK', lmax=lmax_out)
    dl_bb_lensed = lensed_cls[:, 2]  # Index 2: BB

    # --- 2. Lensing potential power spectrum ([l(l+1)]^2 C_l^{phiphi} / 2pi) ---
    cl_pp_camb = results.get_lens_potential_cls(lmax=lmax_out)
    cl_pp = cl_pp_camb[:, 0]  # Index 0: PP

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
    # cl_pp_res = cl_pp * (1 - (cl_pp / (cl_pp + n0)))
    f_filter = np.zeros_like(cl_pp)
    valid_mask = (cl_pp + n0_array > 0) & np.isfinite(n0_array)
    f_filter[valid_mask] = cl_pp[valid_mask] / (cl_pp[valid_mask] + n0_array[valid_mask])
    perfect_recon_mask = (n0_array == 0) & (cl_pp > 0)
    f_filter[perfect_recon_mask] = 1.0
    cl_pp_res = cl_pp * (1.0 - f_filter)
    cl_pp_res[0:2] = 0.0  # Set l=0,1 to zero

    # --- 5. Pad the residual lensing potential array to CAMB's internal max_l ---
    cl_pp_res_padded = np.zeros(pars.max_l + 1)
    len_to_copy = min(len(cl_pp_res), len(cl_pp_res_padded))
    cl_pp_res_padded[:len_to_copy] = cl_pp_res[:len_to_copy]

    # --- 6. Delensed B-mode power spectrum (D_l^BB_delensed) [uK^2] ---
    delensed_cls = results.get_lensed_cls_with_spectrum(
        clpp=cl_pp_res_padded, CMB_unit='muK', lmax=lmax_out
    )
    dl_bb_delensed = delensed_cls[:, 2]  # Index 2: BB

    # --- 7. Delensing efficiency for l=2 to l=100 ---
    ls_for_eff = np.arange(2, 101)
    efficiency = np.zeros_like(ls_for_eff, dtype=float)
    lensed_vals = dl_bb_lensed[ls_for_eff]
    delensed_vals = dl_bb_delensed[ls_for_eff]
    mask_lensed_gt_zero = lensed_vals > 1e-30
    efficiency[mask_lensed_gt_zero] = 100.0 * (lensed_vals[mask_lensed_gt_zero] - delensed_vals[mask_lensed_gt_zero]) / lensed_vals[mask_lensed_gt_zero]

    # --- 8. Save results ---
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "result.csv")
    output_df = pd.DataFrame({'l': ls_for_eff, 'delensing_efficiency': efficiency})
    output_df.to_csv(output_path, index=False)

    # --- Print detailed summary ---
    pd.set_option('display.precision', 6)
    pd.set_option('display.max_rows', 100)
    print("\nDelensing efficiency results (l=2 to l=100):")
    print(output_df)
    print("\nCalculation complete. Results saved to " + output_path)
    print("All power spectra are in micro-Kelvin squared (uK^2).")
    print("Delensing efficiency is in percent (%).")
    print("Sample values:")
    for idx in [0, 24, 49, 74, 98]:  # l=2, 26, 51, 76, 100
        l_val = ls_for_eff[idx]
        print("l = " + str(l_val) + ": lensed BB = " + str(dl_bb_lensed[l_val]) + " uK^2, delensed BB = " + str(dl_bb_delensed[l_val]) + " uK^2, efficiency = " + str(efficiency[idx]) + " %")

if __name__ == "__main__":
    main()