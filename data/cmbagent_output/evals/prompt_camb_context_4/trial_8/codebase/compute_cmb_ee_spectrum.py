# filename: codebase/compute_cmb_ee_spectrum.py
import camb
import numpy as np
import os

def compute_cmb_ee_spectrum(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density [dimensionless]
    omch2=0.122,            # Cold dark matter density [dimensionless]
    mnu=0.06,               # Neutrino mass sum [eV]
    omk=0.0,                # Curvature [dimensionless]
    tau=0.04,               # Optical depth to reionization [dimensionless]
    As=2.0e-9,              # Scalar amplitude [dimensionless]
    ns=0.965,               # Scalar spectral index [dimensionless]
    lmin=2,                 # Minimum multipole
    lmax=3000,              # Maximum multipole
    output_folder="data/",  # Output folder for results
    output_filename="result.csv" # Output CSV filename
):
    r"""
    Compute the CMB E-mode polarization power spectrum D_ell^{EE} = ell(ell+1)C_ell^{EE}/(2pi)
    in units of microKelvin^2 for a flat Lambda CDM cosmology using CAMB.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter (Omega_b h^2).
    omch2 : float
        Physical cold dark matter density parameter (Omega_c h^2).
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter (Omega_k).
    tau : float
        Optical depth to reionization.
    As : float
        Scalar amplitude of primordial power spectrum.
    ns : float
        Scalar spectral index.
    lmin : int
        Minimum multipole moment to include in output.
    lmax : int
        Maximum multipole moment to include in output.
    output_folder : str
        Folder to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
        Saves the results to a CSV file with columns 'l' and 'EE' (in microKelvin^2).
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Set up CAMB parameters
    pars = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu,
        omk=omk,
        tau=tau,
        As=As,
        ns=ns,
        lmax=lmax,
        lens_potential_accuracy=1,
        WantTensors=False
    )

    # Run CAMB and get results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in muK^2, D_ell = l(l+1)C_ell/(2pi)
    powers = results.get_cmb_power_spectra(
        pars,
        CMB_unit='muK',
        raw_cl=False,
        spectra=['lensed_scalar']
    )
    lensed_scalar_cls = powers['lensed_scalar']  # shape: (lmax+1, 4)

    # Extract EE spectrum (column 1), for l=lmin to lmax
    ls = np.arange(lensed_scalar_cls.shape[0])  # l = 0, 1, ..., lmax
    if lmax >= lensed_scalar_cls.shape[0]:
        print("Warning: Requested lmax exceeds available data. Truncating to lmax = " + str(lensed_scalar_cls.shape[0] - 1))
        lmax = lensed_scalar_cls.shape[0] - 1
    l_slice = slice(lmin, lmax + 1)
    output_ls = ls[l_slice]
    output_EE = lensed_scalar_cls[l_slice, 1]  # EE column

    # Save to CSV
    output_data = np.column_stack((output_ls, output_EE))
    csv_path = os.path.join(output_folder, output_filename)
    header = "l,EE"
    np.savetxt(csv_path, output_data, delimiter=",", header=header, comments='')

    # Print summary and first few rows
    print("CMB E-mode polarization power spectrum (D_ell^{EE} = l(l+1)C_ell^{EE}/(2pi)) computed.")
    print("Cosmological parameters:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  Sum m_nu = " + str(mnu) + " eV")
    print("  Omega_k = " + str(omk))
    print("  tau = " + str(tau))
    print("  A_s = " + str(As))
    print("  n_s = " + str(ns))
    print("Multipole range: l = " + str(output_ls[0]) + " to " + str(output_ls[-1]))
    print("Results saved to: " + csv_path)
    print("First 5 rows (l, EE [microKelvin^2]):")
    for i in range(min(5, output_data.shape[0])):
        print(str(int(output_data[i, 0])) + ", " + str(output_data[i, 1]))

if __name__ == "__main__":
    compute_cmb_ee_spectrum()
