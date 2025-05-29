# filename: codebase/cmb_tt_spectrum.py
import camb
from camb import model
import numpy as np
import os


def compute_cmb_tt_spectrum(
    H0=67.5,           # Hubble constant [km/s/Mpc]
    ombh2=0.02,        # Baryon density Omega_b h^2 [dimensionless]
    omch2=0.122,       # Cold dark matter density Omega_c h^2 [dimensionless]
    mnu=0.06,          # Neutrino mass sum [eV]
    omk=0.0,           # Curvature Omega_k [dimensionless]
    tau=0.06,          # Optical depth to reionization [dimensionless]
    As=2e-9,           # Scalar amplitude [dimensionless]
    ns=0.965,          # Scalar spectral index [dimensionless]
    lmax=3000,         # Maximum multipole [dimensionless]
    output_folder="data",
    output_filename="result.csv"
):
    r"""
    Compute the CMB temperature power spectrum l(l+1)C_l^{TT}/(2pi) in units of muK^2
    for a flat Lambda CDM cosmology using CAMB, and save the results to a CSV file.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density Omega_b h^2 (dimensionless).
    omch2 : float
        Physical cold dark matter density Omega_c h^2 (dimensionless).
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter Omega_k (dimensionless).
    tau : float
        Optical depth to reionization (dimensionless).
    As : float
        Scalar amplitude (dimensionless).
    ns : float
        Scalar spectral index (dimensionless).
    lmax : int
        Maximum multipole moment to compute (dimensionless).
    output_folder : str
        Directory to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
        The function saves the results to a CSV file and prints a summary.
    """
    # Ensure output directory exists
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
        lmax=lmax
    )
    # Use non-linear corrections for lensing
    pars.NonLinear = model.NonLinear_both

    # Run CAMB
    results = camb.get_results(pars)

    # Get lensed CMB power spectra in muK^2
    powers = results.get_cmb_power_spectra(CMB_unit='muK')
    total_cls = powers['total']  # shape (lmax+1, 4): columns are TT, EE, BB, TE

    # Extract l=2..lmax TT spectrum
    l = np.arange(2, lmax + 1)  # [dimensionless]
    TT = total_cls[2:lmax + 1, 0]  # [muK^2]

    # Save to CSV
    output_path = os.path.join(output_folder, output_filename)
    np.savetxt(output_path, np.column_stack([l, TT]), delimiter=',', header='l,TT', comments='')

    # Print summary
    print("CMB TT power spectrum calculation complete.")
    print("Results saved to " + output_path)
    print("Columns: l (multipole, dimensionless), TT (l(l+1)C_l^{TT}/(2pi) in muK^2)")
    print("First 10 rows:")
    for i in range(min(10, len(l))):
        print("l = " + str(int(l[i])) + ", TT = " + str(TT[i]) + " muK^2")
    print("Total number of rows: " + str(len(l)))


if __name__ == "__main__":
    compute_cmb_tt_spectrum()