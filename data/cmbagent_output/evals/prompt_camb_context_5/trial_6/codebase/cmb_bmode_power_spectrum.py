# filename: codebase/cmb_bmode_power_spectrum.py
import camb
import numpy as np
import os

def compute_cmb_bmode_power_spectrum():
    r"""
    Compute the CMB B-mode polarization power spectrum (l(l+1)C_l^{BB}/(2pi)) for a flat Lambda CDM cosmology
    using CAMB, with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (Omega_b h^2): 0.022
        - Cold dark matter density (Omega_c h^2): 0.122
        - Neutrino mass sum (Sigma m_nu): 0.06 eV
        - Curvature (Omega_k): 0
        - Optical depth to reionization (tau): 0.06
        - Tensor-to-scalar ratio (r): 0
        - Scalar amplitude (A_s): 2e-9
        - Scalar spectral index (n_s): 0.965

    The B-mode power spectrum is computed for multipole moments l=2 to l=3000, in units of micro-Kelvin squared (uK^2).
    The results are saved to 'data/result.csv' with columns:
        l: Multipole moment (integer, 2 to 3000)
        BB: B-mode polarization power spectrum (l(l+1)C_l^{BB}/(2pi), in uK^2)
    """
    # Cosmological parameters
    H0 = 67.5  # [km/s/Mpc]
    ombh2 = 0.022  # [dimensionless]
    omch2 = 0.122  # [dimensionless]
    mnu = 0.06  # [eV]
    omk = 0.0  # [dimensionless]
    tau = 0.06  # [dimensionless]
    As = 2.0e-9  # [dimensionless]
    ns = 0.965  # [dimensionless]
    r_tensor = 0.0  # [dimensionless]
    l_max_calc = 3000  # [dimensionless]
    lens_acc = 1  # [dimensionless]

    # Initialize CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=r_tensor)
    pars.set_for_lmax(lmax=l_max_calc, lens_potential_accuracy=lens_acc)

    # Run CAMB calculation
    results = camb.get_results(pars)

    # Extract lensed scalar CMB power spectra (Dl = l(l+1)Cl/2pi, in uK^2)
    lensed_cls = results.get_lensed_scalar_cls(CMB_unit='muK', lmax=l_max_calc)
    # lensed_cls shape: (l_max_calc+1, 4), columns: TT, EE, BB, TE

    # Prepare l and BB arrays for l=2..l_max_calc
    ls = np.arange(2, l_max_calc + 1)  # [dimensionless]
    bb_spectrum = lensed_cls[2:l_max_calc + 1, 2]  # [uK^2]

    # Stack and save to CSV
    output_data = np.column_stack((ls, bb_spectrum))
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, "result.csv")
    header_text = "l,BB"
    np.savetxt(output_filename, output_data, delimiter=',', header=header_text, comments='', fmt=['%d', '%.18e'])

    # Print summary to console
    print("CMB B-mode polarization power spectrum (l(l+1)C_l^{BB}/(2pi)) computed for l=2 to l=3000.")
    print("Units: BB in micro-Kelvin squared (uK^2).")
    print("Results saved to " + output_filename)
    print("\nFirst 5 rows of the output file:")
    print(header_text)
    for i in range(min(5, len(output_data))):
        print(str(int(output_data[i, 0])) + "," + "%.6e" % output_data[i, 1])


if __name__ == "__main__":
    compute_cmb_bmode_power_spectrum()