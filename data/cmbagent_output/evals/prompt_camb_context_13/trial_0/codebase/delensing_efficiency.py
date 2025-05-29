# filename: codebase/delensing_efficiency.py
import camb
import numpy as np
import pandas as pd
import os

def main():
    r"""
    Calculate the delensing efficiency of the CMB B-mode polarization power spectrum for a flat Lambda CDM cosmology
    using CAMB and a provided lensing noise power spectrum.

    Steps:
    1. Set up cosmological parameters in CAMB.
    2. Run CAMB to obtain lensed B-mode and lensing potential power spectra up to l=2000.
    3. Load the lensing noise power spectrum (N0) from a CSV file and align it with CAMB output arrays.
    4. Compute the residual lensing potential power spectrum.
    5. Pad the residual lensing potential to the length required by CAMB.
    6. Compute the delensed B-mode power spectrum using the residual lensing potential.
    7. Calculate the delensing efficiency for l=2 to 100.
    8. Save the results in 'data/result.csv' and print a detailed summary to the console.
    """

    # --- Cosmological Parameters (units in comments) ---
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    As = 2.0e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]

    lmax_calc = 3000  # Max ell for CAMB's internal calculation (higher for accuracy)
    lens_potential_accuracy_val = 2  # Accuracy for lensing calculation
    lmax_out = 2000  # Max ell for output spectra (C_l^BB, C_l^phiphi)

    # --- Setup CAMB ---
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax_calc, lens_potential_accuracy=lens_potential_accuracy_val)
    pars.WantTensors = False  # Only scalar-induced B-modes (lensing)

    # --- Get CAMB Results ---
    results = camb.get_results(pars)

    # --- 1. Lensed B-mode power spectrum (D_ell^BB) in muK^2 ---
    lensed_cls = results.get_lensed_scalar_cls(CMB_unit='muK', lmax=lmax_out)
    dl_bb_lensed = lensed_cls[:, 2]  # D_ell^BB, units: muK^2

    # --- 2. CMB lensing potential power spectrum (cl_pp) ---
    cl_pp_camb_output = results.get_lens_potential_cls(lmax=lmax_out)
    cl_pp = cl_pp_camb_output[:, 0]  # [L(L+1)]^2 C_L^phiphi / (2pi), units: dimensionless

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

    # --- 5. Pad the residual lensing potential array ---
    cl_pp_res_padded = np.zeros(pars.max_l + 1)
    len_to_copy = min(len(cl_pp_res), len(cl_pp_res_padded))
    cl_pp_res_padded[:len_to_copy] = cl_pp_res[:len_to_copy]

    # --- 6. Delensed B-mode power spectrum (D_ell^BB_delensed) ---
    delensed_cls_output = results.get_lensed_cls_with_spectrum(
        clpp=cl_pp_res_padded, CMB_unit='muK', lmax=lmax_out
    )
    dl_bb_delensed = delensed_cls_output[:, 2]  # D_ell^BB_delensed, units: muK^2

    # --- 7. Delensing efficiency ---
    ls_for_eff = np.arange(2, 101)  # Multipoles for efficiency calculation
    efficiency = np.zeros_like(ls_for_eff, dtype=float)
    mask_lensed_gt_zero = dl_bb_lensed[ls_for_eff] > 1e-30
    efficiency[mask_lensed_gt_zero] = 100.0 * (
        dl_bb_lensed[ls_for_eff][mask_lensed_gt_zero] - dl_bb_delensed[ls_for_eff][mask_lensed_gt_zero]
    ) / dl_bb_lensed[ls_for_eff][mask_lensed_gt_zero]

    # --- 8. Save results ---
    if not os.path.exists('data'):
        os.makedirs('data')
    output_df = pd.DataFrame({'l': ls_for_eff, 'delensing_efficiency': efficiency})
    output_csv_path = os.path.join('data', 'result.csv')
    output_df.to_csv(output_csv_path, index=False)

    # --- Print detailed summary ---
    pd.set_option('display.float_format', lambda x: '%.6e' % x)
    print("\nDelensing efficiency results (l=2 to l=100):")
    print(output_df)
    print("\nCalculation complete. Results saved to " + output_csv_path)
    print("\nSample values:")
    for l in [2, 10, 50, 100]:
        idx = l - 2
        print(
            "l = " + str(l) +
            ", lensed BB = " + str(dl_bb_lensed[l]) + " muK^2" +
            ", delensed BB = " + str(dl_bb_delensed[l]) + " muK^2" +
            ", efficiency = " + str(efficiency[idx]) + " %"
        )

if __name__ == "__main__":
    main()