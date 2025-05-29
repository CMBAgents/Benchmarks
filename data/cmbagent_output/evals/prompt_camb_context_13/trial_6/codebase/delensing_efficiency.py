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
    2. Run CAMB to obtain lensed B-mode and lensing potential power spectra up to l=2000.
    3. Load external lensing noise power spectrum from CSV and align with CAMB output.
    4. Compute the residual lensing potential power spectrum.
    5. Pad the residual spectrum to CAMB's internal calculation length.
    6. Use the residual lensing potential to compute the delensed B-mode power spectrum.
    7. Calculate delensing efficiency for l=2 to l=100.
    8. Save results to 'data/result.csv' and print a summary.

    All power spectra are in units of microkelvin squared (muK^2) where appropriate.
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

    lmax_out = 2000  # Maximum multipole for output [dimensionless]
    lmax_camb = 3000  # Internal CAMB lmax for accuracy [dimensionless]
    lens_potential_accuracy = 2  # Lensing accuracy (higher is more accurate)

    # --- Setup CAMB ---
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax_camb, lens_potential_accuracy=lens_potential_accuracy)
    pars.WantTensors = False  # Only scalar-induced B-modes

    # --- Run CAMB ---
    results = camb.get_results(pars)

    # --- 1. Lensed B-mode power spectrum (D_ell^BB) [muK^2] ---
    lensed_cls = results.get_lensed_scalar_cls(CMB_unit='muK', lmax=lmax_out)
    dl_bb_lensed = lensed_cls[:, 2]  # [muK^2], index 2 is BB

    # --- 2. CMB lensing potential power spectrum ([l(l+1)]^2 C_l^{phiphi}/2pi) ---
    cl_pp_camb = results.get_lens_potential_cls(lmax=lmax_out)
    cl_pp = cl_pp_camb[:, 0]  # [l(l+1)]^2 C_l^{phiphi}/2pi

    # --- 3. Load N0 and align arrays ---
    n0_file_path = '/Users/antoidicherianlonappan/Software/Benchmarks/data/N0.csv'
    n0_data = pd.read_csv(n0_file_path)
    n0_array = np.full(lmax_out + 1, np.inf)
    file_ls = n0_data['l'].values.astype(int)
    file_Nls = n0_data['Nl'].values
    mask = (file_ls >= 0) & (file_ls <= lmax_out)
    n0_array[file_ls[mask]] = file_Nls[mask]

    # --- 4. Residual lensing potential power spectrum ---
    # cl_pp_res = cl_pp * (1 - (cl_pp / (cl_pp + n0)))
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = cl_pp + n0_array
        ratio = np.zeros_like(cl_pp)
        valid = denom > 0
        ratio[valid] = cl_pp[valid] / denom[valid]
        cl_pp_res = cl_pp * (1.0 - ratio)
    cl_pp_res[0:2] = 0.0  # Set l=0,1 to zero

    # --- 5. Pad the residual lensing potential array to CAMB's max_l ---
    cl_pp_res_padded = np.zeros(pars.max_l + 1)
    ncopy = min(len(cl_pp_res), len(cl_pp_res_padded))
    cl_pp_res_padded[:ncopy] = cl_pp_res[:ncopy]

    # --- 6. Delensed B-mode power spectrum (D_ell^BB,delensed) [muK^2] ---
    delensed_cls = results.get_lensed_cls_with_spectrum(
        clpp=cl_pp_res_padded, CMB_unit='muK', lmax=lmax_out
    )
    dl_bb_delensed = delensed_cls[:, 2]  # [muK^2], index 2 is BB

    # --- 7. Delensing efficiency for l=2..100 ---
    ls = np.arange(2, 101)
    efficiency = np.zeros_like(ls, dtype=float)
    lensed_vals = dl_bb_lensed[ls]
    delensed_vals = dl_bb_delensed[ls]
    mask = lensed_vals > 1e-30
    efficiency[mask] = 100.0 * (lensed_vals[mask] - delensed_vals[mask]) / lensed_vals[mask]

    # --- 8. Save results ---
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'l': ls, 'delensing_efficiency': efficiency})
    df.to_csv(output_path, index=False)

    # --- Print summary ---
    np.set_printoptions(precision=4, suppress=True)
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
    print("\nDelensing efficiency (percent) for l=2..100:\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()