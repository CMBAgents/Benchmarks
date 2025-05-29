# filename: codebase/cmb_tt_spectrum.py
import camb
from camb import model
import numpy as np
import os

def compute_cmb_tt_spectrum(
    H0=67.5,           # Hubble constant [km/s/Mpc]
    ombh2=0.02,        # Baryon density parameter [dimensionless]
    omch2=0.122,       # Cold dark matter density parameter [dimensionless]
    mnu=0.06,          # Sum of neutrino masses [eV]
    omk=0.0,           # Curvature parameter [dimensionless]
    tau=0.06,          # Optical depth to reionization [dimensionless]
    As=2e-9,           # Scalar amplitude [dimensionless]
    ns=0.965,          # Scalar spectral index [dimensionless]
    lmax=3000,         # Maximum multipole [dimensionless]
    output_dir="data", # Output directory for results
    output_filename="result.csv" # Output CSV filename
):
    r"""
    Compute the CMB temperature power spectrum l(l+1)C_l^{TT}/(2\pi) in units of microKelvin^2
    for a flat Lambda CDM cosmology using CAMB, and save the results to a CSV file.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter, Omega_b h^2.
    omch2 : float
        Physical cold dark matter density parameter, Omega_c h^2.
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter, Omega_k.
    tau : float
        Optical depth to reionization.
    As : float
        Scalar amplitude of primordial power spectrum.
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
        The function saves the computed spectrum to a CSV file and prints a summary.
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
        lmax=lmax
    )
    # Use non-linear corrections for lensing
    pars.NonLinear = model.NonLinear_both

    # Run CAMB
    results = camb.get_results(pars)

    # Get lensed CMB power spectra in microKelvin^2
    powers = results.get_cmb_power_spectra(CMB_unit='muK')
    total_cls = powers['total']  # shape (lmax+1, 4): columns are TT, EE, BB, TE

    # Extract l=2..lmax for TT spectrum
    l_vals = np.arange(2, lmax + 1)  # Multipole moments [dimensionless]
    tt_vals = total_cls[2:lmax + 1, 0]  # TT spectrum [microKelvin^2]

    # Prepare output array
    output_data = np.column_stack((l_vals, tt_vals))

    # Save to CSV
    output_path = os.path.join(output_dir, output_filename)
    np.savetxt(output_path, output_data, delimiter=',', header='l,TT', comments='')

    # Print summary
    print("CMB TT power spectrum computed for flat Lambda CDM cosmology.")
    print("Cosmological parameters used:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  Sum m_nu = " + str(mnu) + " eV")
    print("  Omega_k = " + str(omk))
    print("  tau = " + str(tau))
    print("  A_s = " + str(As))
    print("  n_s = " + str(ns))
    print("  lmax = " + str(lmax))
    print("Results saved to: " + output_path)
    print("First 5 rows (l, TT [microKelvin^2]):")
    np.set_printoptions(precision=6, suppress=True)
    print(output_data[:5])


if __name__ == "__main__":
    compute_cmb_tt_spectrum()