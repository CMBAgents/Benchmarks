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
    1. Set cosmological parameters and configure CAMB for accurate lensing and B-mode spectra.
    2. Compute the lensed B-mode power spectrum (C_ell^BB) and lensing potential power spectrum (C_ell^phiphi).
    3. Load the lensing noise power spectrum (N0) from a CSV file and align it with the lensing potential array.
    4. Calculate the residual lensing potential power spectrum.
    5. Pad the residual lensing potential array to match CAMB's internal maximum multipole.
    6. Compute the delensed B-mode power spectrum using the residual lensing potential.
    7. Calculate the delensing efficiency.
    8. Save the results for l=2 to 100 in a CSV file under the data/ directory and print a detailed summary.

    Units:
    - All CMB power spectra are in micro-Kelvin squared (muK^2).
    - Lensing potential spectra are in [L(L+1)]^2 C_L^{phiphi}/2pi units.
    """
    # --- Cosmological Parameters ---
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    As = 2e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]

    lmax_requested = 2000  # Maximum multipole for output spectra
    lmax_internal = 3000  # Internal CAMB lmax for accuracy
    lens_potential_accuracy = 2  # For accurate lensing and BB

    # --- Set up CAMB parameters ---
    pars = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu,
        omk=omk,
        tau=tau,
        As=As,
        ns=ns,
        WantTensors=False,
        AccuracyBoost=1,
        lSampleBoost=1,
        lAccuracyBoost=1
    )
    pars.set_for_lmax(lmax_internal, lens_potential_accuracy=lens_potential_accuracy)
    pars.NonLinear = model.NonLinear_lens

    # --- Run CAMB ---
    results = camb.get_results(pars)

    # --- 1. Lensed B-mode power spectrum (C_ell^BB) ---
    lensed_cls = results.get_lensed_scalar_cls(CMB_unit='muK', lmax=lmax_requested)
    cl_bb_lensed = lensed_cls[:lmax_requested + 1, 2]  # [muK^2]

    # --- 2. CMB lensing potential power spectrum (C_ell^phiphi) ---
    lens_potential_cls = results.get_lens_potential_cls(lmax=lmax_requested)
    cl_pp = lens_potential_cls[:lmax_requested + 1, 0]  # [L(L+1)]^2 C_L^{phiphi}/2pi

    # --- 3. Load lensing noise power spectrum (N0) and align arrays ---
    n0_file_path = '/Users/antoidicherianlonappan/Software/Benchmarks/data/N0.csv'
    n0_data = pd.read_csv(n0_file_path)
    n0_array = np.full(lmax_requested + 1, np.inf)
    valid_ells_mask = (n0_data['l'] >= 0) & (n0_data['l'] <= lmax_requested)
    ells_in_file = n0_data.loc[valid_ells_mask, 'l'].astype(int).values
    Nl_in_file = n0_data.loc[valid_ells_mask, 'Nl'].values
    n0_array[ells_in_file] = Nl_in_file

    # --- 4. Residual lensing potential power spectrum ---
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.divide(cl_pp, cl_pp + n0_array, out=np.zeros_like(cl_pp), where=(cl_pp + n0_array) != 0)
    cl_pp_res = cl_pp * (1 - term)
    cl_pp_res[0] = 0.0
    cl_pp_res[1] = 0.0

    # --- 5. Pad the residual lensing potential array ---
    internal_camb_lmax = results.Params.max_l
    cl_pp_res_padded = np.zeros(internal_camb_lmax + 1)
    num_elements_to_copy = min(len(cl_pp_res), internal_camb_lmax + 1)
    cl_pp_res_padded[:num_elements_to_copy] = cl_pp_res[:num_elements_to_copy]

    # --- 6. Delensed B-mode power spectrum (C_ell^BB,delensed) ---
    delensed_cls = results.get_lensed_cls_with_spectrum(
        clpp=cl_pp_res_padded,
        lmax=lmax_requested,
        CMB_unit='muK'
    )
    cl_bb_delensed = delensed_cls[:lmax_requested + 1, 2]  # [muK^2]

    # --- 7. Delensing efficiency ---
    efficiency = np.zeros_like(cl_bb_lensed)
    mask_l_ge_2 = np.arange(len(cl_bb_lensed)) >= 2
    mask_nonzero_lensed_bb = (cl_bb_lensed != 0) & mask_l_ge_2
    efficiency[mask_nonzero_lensed_bb] = 100.0 * (
        cl_bb_lensed[mask_nonzero_lensed_bb] - cl_bb_delensed[mask_nonzero_lensed_bb]
    ) / cl_bb_lensed[mask_nonzero_lensed_bb]

    # --- 8. Save results to CSV (l=2 to 100) ---
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

    # --- Print detailed summary ---
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print("Delensing efficiency calculation complete.\n")
    print("Cosmological parameters used:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  ombh2 = " + str(ombh2))
    print("  omch2 = " + str(omch2))
    print("  mnu = " + str(mnu) + " eV")
    print("  omk = " + str(omk))
    print("  tau = " + str(tau))
    print("  As = " + str(As))
    print("  ns = " + str(ns))
    print("\nResults saved to: " + output_path)
    print("\nSample of delensing efficiency (l=2 to l=10):")
    for l in range(2, 11):
        print("  l = " + str(l) + " : " + str(efficiency[l]) + " %")
    print("\nFull results for l=2 to l=100 are available in the CSV file.")


if __name__ == "__main__":
    main()
