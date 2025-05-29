# filename: codebase/cmb_bmode_spectrum.py
import camb
import numpy as np
import os


def compute_cmb_bmode_spectrum(
    H0=67.5,           # Hubble constant [km/s/Mpc]
    ombh2=0.022,       # Baryon density Omega_b h^2 [dimensionless]
    omch2=0.122,       # Cold dark matter density Omega_c h^2 [dimensionless]
    mnu=0.06,          # Sum of neutrino masses [eV]
    omk=0.0,           # Curvature Omega_k [dimensionless]
    tau=0.06,          # Optical depth to reionization
    As=2.0e-9,         # Scalar amplitude
    ns=0.965,          # Scalar spectral index
    r_tensor=0.0,      # Tensor-to-scalar ratio
    l_max_calc=3000,   # Maximum multipole [dimensionless]
    lens_acc=1,        # Lensing potential accuracy [integer]
    output_dir="data", # Output directory [string]
    output_filename="result.csv" # Output CSV filename [string]
):
    """
    Compute the lensed CMB B-mode polarization power spectrum for a flat Lambda CDM cosmology
    and save the results to a CSV file.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density Omega_b h^2.
    omch2 : float
        Physical cold dark matter density Omega_c h^2.
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter Omega_k.
    tau : float
        Optical depth to reionization.
    As : float
        Scalar amplitude.
    ns : float
        Scalar spectral index.
    r_tensor : float
        Tensor-to-scalar ratio.
    l_max_calc : int
        Maximum multipole moment to compute.
    lens_acc : int
        Lensing potential accuracy.
    output_dir : str
        Directory to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
        Saves the results to a CSV file.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Initialize CAMB parameters
    pars = camb.CAMBparams()

    # 2. Set cosmological parameters
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)

    # 3. Set initial power spectrum parameters
    pars.InitPower.set_params(As=As, ns=ns, r=r_tensor)

    # 4. Configure calculation settings (Lmax and lensing accuracy)
    pars.set_for_lmax(lmax=l_max_calc, lens_potential_accuracy=lens_acc)

    # 5. Perform the CAMB calculation
    results = camb.get_results(pars)

    # 6. Extract the B-mode power spectrum
    # get_lensed_scalar_cls returns Dl = l(l+1)Cl/2pi in μK^2 if CMB_unit='muK'
    # Columns: 0=TT, 1=EE, 2=BB, 3=TE
    all_cls = results.get_lensed_scalar_cls(CMB_unit='muK', lmax=l_max_calc)
    ls = np.arange(2, l_max_calc + 1)  # Multipole moments [dimensionless]
    bb_spectrum = all_cls[2:l_max_calc + 1, 2]  # B-mode spectrum [μK^2]

    # 7. Prepare data for CSV
    output_data = np.column_stack((ls, bb_spectrum))

    # 8. Save to CSV
    output_path = os.path.join(output_dir, output_filename)
    np.savetxt(output_path, output_data, delimiter=',' , header='l,BB', comments='', fmt=['%d', '%.18e'])

    # Print summary to console
    print("CMB B-mode polarization power spectrum (l(l+1)C_l^{BB}/(2pi)) computed for l=2..3000.")
    print("Results saved to " + output_path)
    print("Columns: l (multipole moment), BB (μK^2)")
    print("First 5 rows:")
    print("l,BB")
    for i in range(min(5, output_data.shape[0])):
        print(str(int(output_data[i, 0])) + "," + "{:.6e}".format(output_data[i, 1]))


if __name__ == "__main__":
    compute_cmb_bmode_spectrum()