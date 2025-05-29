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
    lmax=3000,              # Maximum multipole moment [dimensionless]
    output_dir="data/",     # Output directory for result file
    output_filename="result.csv" # Output CSV filename
):
    r"""
    Compute the CMB E-mode polarization power spectrum D_l^{EE} = l(l+1)C_l^{EE}/(2pi)
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
        Scalar amplitude of primordial fluctuations.
    ns : float
        Scalar spectral index.
    lmax : int
        Maximum multipole moment to compute.
    output_dir : str
        Directory to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
        Saves the result as a CSV file with columns:
        l: Multipole moment (integer, 2 to lmax)
        EE: E-mode polarization power spectrum (D_l^{EE} in microKelvin^2)
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    # Compute results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in muK^2, D_l = l(l+1)C_l/(2pi)
    powers = results.get_cmb_power_spectra(
        pars,
        CMB_unit='muK',
        raw_cl=False,
        spectra=['lensed_scalar']
    )
    lensed_scalar_cls = powers['lensed_scalar']  # shape: (lmax+1, 4)

    # Extract EE spectrum (column 1), for l=2 to lmax
    ls = np.arange(lensed_scalar_cls.shape[0])  # l = 0, 1, ..., lmax
    start_l = 2
    end_l = lmax
    EE_spectrum = lensed_scalar_cls[start_l:end_l+1, 1]
    ls_out = ls[start_l:end_l+1]

    # Stack and save to CSV
    output_data = np.column_stack((ls_out, EE_spectrum))
    output_path = os.path.join(output_dir, output_filename)
    header = "l,EE"
    np.savetxt(output_path, output_data, delimiter=",", header=header, comments='')

    # Print summary to console
    print("CMB E-mode polarization power spectrum (D_l^{EE}) computed and saved to " + output_path)
    print("Columns: l (multipole moment, 2 to " + str(lmax) + "), EE (D_l^{EE} in microKelvin^2)")
    print("First 5 rows:")
    for i in range(min(5, output_data.shape[0])):
        print("l = " + str(int(output_data[i,0])) + ", EE = " + str(output_data[i,1]) + " microKelvin^2")
    print("Total rows saved: " + str(output_data.shape[0]))


if __name__ == "__main__":
    compute_cmb_ee_spectrum()