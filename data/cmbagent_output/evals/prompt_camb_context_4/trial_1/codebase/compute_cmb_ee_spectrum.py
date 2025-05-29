# filename: codebase/compute_cmb_ee_spectrum.py
import camb
import numpy as np
import os

def compute_cmb_ee_spectrum():
    r"""
    Compute the CMB E-mode polarization power spectrum D_ell^{EE} = ell(ell+1)C_ell^{EE}/(2pi)
    in units of microKelvin^2 for a flat Lambda CDM cosmology using CAMB, for multipoles
    ell=2 to 3000, and save the results to 'data/result.csv'.

    Cosmological parameters:
        H0: Hubble constant [km/s/Mpc]
        ombh2: Physical baryon density [dimensionless]
        omch2: Physical cold dark matter density [dimensionless]
        mnu: Sum of neutrino masses [eV]
        omk: Curvature density [dimensionless]
        tau: Optical depth to reionization [dimensionless]
        As: Scalar amplitude [dimensionless]
        ns: Scalar spectral index [dimensionless]

    Output:
        CSV file with columns:
            l: Multipole moment (integer, 2 to 3000)
            EE: E-mode polarization power spectrum D_ell^{EE} [microKelvin^2]
    """
    # --- Cosmological Parameters ---
    H0 = 67.5  # [km/s/Mpc]
    ombh2 = 0.022  # [dimensionless]
    omch2 = 0.122  # [dimensionless]
    mnu = 0.06  # [eV]
    omk = 0.0  # [dimensionless]
    tau = 0.04  # [dimensionless]
    As = 2.0e-9  # [dimensionless]
    ns = 0.965  # [dimensionless]

    # --- Calculation Settings ---
    lmax_calc = 3000  # [dimensionless]
    lens_potential_accuracy_setting = 1  # [dimensionless]

    # 1. Set up CAMB parameters
    pars = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu,
        omk=omk,
        tau=tau,
        As=As,
        ns=ns,
        lmax=lmax_calc,
        lens_potential_accuracy=lens_potential_accuracy_setting,
        WantTensors=False
    )

    # 2. Get results
    results = camb.get_results(pars)

    # 3. Extract CMB power spectra
    powers = results.get_cmb_power_spectra(
        pars,
        CMB_unit='muK',
        raw_cl=False,
        spectra=['lensed_scalar']
    )
    lensed_scalar_cls = powers['lensed_scalar']
    EE_spectrum_full = lensed_scalar_cls[:, 1]  # [microKelvin^2]

    # 4. Format and save data
    ls_full = np.arange(EE_spectrum_full.shape[0])  # [dimensionless]
    start_l = 2
    end_l = 3000
    output_ls = ls_full[start_l:end_l+1]
    output_EE_spectrum = EE_spectrum_full[start_l:end_l+1]
    output_data = np.column_stack((output_ls, output_EE_spectrum))

    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_filename = os.path.join(output_dir, "result.csv")
    csv_header = "l,EE"  # l: Multipole moment, EE: E-mode D_l in microKelvin^2
    np.savetxt(csv_filename, output_data, delimiter=',', header=csv_header, comments='')

    # Print summary to console
    print("CMB E-mode polarization power spectrum (D_l^EE) computed and saved to " + csv_filename)
    print("Columns: l (multipole, 2-3000), EE (D_l^EE in microKelvin^2)")
    print("First 5 rows:")
    for i in range(min(5, output_data.shape[0])):
        print("l = " + str(int(output_data[i, 0])) + ", EE = " + str(output_data[i, 1]) + " microKelvin^2")
    print("Total rows saved: " + str(output_data.shape[0]))

if __name__ == "__main__":
    compute_cmb_ee_spectrum()