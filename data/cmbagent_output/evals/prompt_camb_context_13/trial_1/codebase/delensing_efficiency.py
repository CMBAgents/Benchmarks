# filename: codebase/delensing_efficiency.py
import camb
from camb import model
import numpy as np
import pandas as pd
import os

def main():
    r"""
    Calculate the delensing efficiency of the CMB B-mode polarization power spectrum
    for a flat Lambda CDM cosmology using CAMB and a provided lensing noise power spectrum.

    Steps:
    1. Set up cosmological parameters and CAMB configuration.
    2. Compute lensed B-mode and lensing potential power spectra up to l=2000.
    3. Load and align the lensing noise power spectrum from CSV.
    4. Calculate the residual lensing potential power spectrum.
    5. Pad the residual lensing potential for CAMB.
    6. Compute the delensed B-mode power spectrum.
    7. Calculate delensing efficiency.
    8. Save results for l=2 to 100 in a CSV file under 'data/'.

    All power spectra are in units of micro-Kelvin squared (μK^2) where appropriate.
    """
    # --- Cosmological Parameters ---
    H0_val = 67.5  # Hubble constant [km/s/Mpc]
    ombh2_val = 0.022  # Baryon density [dimensionless]
    omch2_val = 0.122  # Cold dark matter density [dimensionless]
    mnu_val = 0.06  # Neutrino mass sum [eV]
    omk_val = 0.0  # Curvature [dimensionless]
    tau_val = 0.06  # Optical depth to reionization [dimensionless]
    As_val = 2e-9  # Scalar amplitude [dimensionless]
    ns_val = 0.965  # Scalar spectral index [dimensionless]

    lmax_requested = 2000  # Max multipole for output spectra
    lmax_calc_internal = 3000  # For accurate BB, must be >= lmax_requested
    lens_potential_accuracy_val = 2  # For accurate lensing and BB

    # --- CAMB Setup ---
    pars = camb.set_params(
        H0=H0_val,
        ombh2=ombh2_val,
        omch2=omch2_val,
        mnu=mnu_val,
        omk=omk_val,
        tau=tau_val,
        As=As_val,
        ns=ns_val,
        WantTensors=False,
        AccuracyBoost=1,
        lSampleBoost=1,
        lAccuracyBoost=1
    )
    pars.set_for_lmax(lmax_calc_internal, lens_potential_accuracy=lens_potential_accuracy_val)
    pars.NonLinear = model.NonLinear_lens

    results = camb.get_results(pars)

    # --- 1. Lensed B-mode power spectrum (C_ell^BB) ---
    lensed_cls_all = results.get_lensed_scalar_cls(CMB_unit='muK', lmax=lmax_requested)
    cl_bb_lensed = lensed_cls_all[:lmax_requested + 1, 2]  # [μK^2]

    # --- 2. CMB lensing potential power spectrum (C_ell^phiphi) ---
    lens_potential_cls_all = results.get_lens_potential_cls(lmax=lmax_requested)
    cl_pp = lens_potential_cls_all[:lmax_requested + 1, 0]  # [(ell(ell+1))^2 C_ell^{phiphi} / (2pi)]

    # --- 3. Load lensing noise power spectrum (N0) and align arrays ---
    n0_file_path = '/Users/antoidicherianlonappan/Software/Benchmarks/data/N0.csv'
    n0_data = pd.read_csv(n0_file_path)
    n0_array = np.full(lmax_requested + 1, np.inf)
    valid_ells_mask = (n0_data['l'] >= 0) & (n0_data['l'] <= lmax_requested)
    ells_in_file = n0_data.loc[valid_ells_mask, 'l'].astype(int).values
    Nl_in_file = n0_data.loc[valid_ells_mask, 'Nl'].values
    n0_array[ells_in_file] = Nl_in_file

    # --- 4. Residual lensing potential power spectrum ---
    # cl_pp_res = cl_pp * (1 - (cl_pp / (cl_pp + n0_array)))
    with np.errstate(divide='ignore', invalid='ignore'):
        term_in_parenthesis = np.divide(cl_pp, cl_pp + n0_array, 
                                       out=np.zeros_like(cl_pp), 
                                       where=(cl_pp + n0_array) != 0)
    cl_pp_res = cl_pp * (1 - term_in_parenthesis)
    cl_pp_res[0] = 0
    cl_pp_res[1] = 0

    # --- 5. Pad the residual lensing potential array ---
    internal_camb_lmax = results.Params.max_l
    cl_pp_res_padded = np.zeros(internal_camb_lmax + 1)
    num_elements_to_copy = min(len(cl_pp_res), internal_camb_lmax + 1)
    cl_pp_res_padded[:num_elements_to_copy] = cl_pp_res[:num_elements_to_copy]

    # --- 6. Delensed B-mode power spectrum (C_ell^BB,delensed) ---
    delensed_cls_all = results.get_lensed_cls_with_spectrum(
        clpp=cl_pp_res_padded,
        lmax=lmax_requested,
        CMB_unit='muK'
    )
    cl_bb_delensed = delensed_cls_all[:lmax_requested + 1, 2]  # [μK^2]

    # --- 7. Delensing efficiency ---
    ls = np.arange(len(cl_bb_lensed))
    efficiency = np.zeros_like(cl_bb_lensed)
    mask_l_ge_2 = ls >= 2
    mask_nonzero_lensed_bb = (cl_bb_lensed != 0) & mask_l_ge_2
    efficiency[mask_nonzero_lensed_bb] = 100 * (cl_bb_lensed[mask_nonzero_lensed_bb] - cl_bb_delensed[mask_nonzero_lensed_bb]) / cl_bb_lensed[mask_nonzero_lensed_bb]

    # --- 8. Save results to CSV for l = 2 to 100 ---
    ls_output = np.arange(2, 101)
    efficiency_output = efficiency[ls_output]
    output_df = pd.DataFrame({
        'l': ls_output,
        'delensing_efficiency': efficiency_output
    })

    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "result.csv")
    output_df.to_csv(output_path, index=False)

    # Print summary to console
    print("Delensing efficiency calculation complete.")
    print("Results saved to " + output_path)
    print("First 10 rows of the result:")
    print(output_df.head(10).to_string(index=False))
    print("\nColumn descriptions:")
    print("l: Multipole moment (integer, 2 <= l <= 100)")
    print("delensing_efficiency: Delensing efficiency in percent (%)")


if __name__ == "__main__":
    main()
