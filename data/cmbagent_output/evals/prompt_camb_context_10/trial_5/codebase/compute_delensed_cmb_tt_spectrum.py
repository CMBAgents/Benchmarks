# filename: codebase/compute_delensed_cmb_tt_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_delensed_cmb_tt_spectrum(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density [dimensionless]
    omch2=0.122,            # Cold dark matter density [dimensionless]
    mnu=0.06,               # Neutrino mass sum [eV]
    omk=0,                  # Curvature [dimensionless]
    tau=0.06,               # Optical depth to reionization [dimensionless]
    As=2e-9,                # Scalar amplitude [dimensionless]
    ns=0.965,               # Scalar spectral index [dimensionless]
    lmax_calc=3500,         # Calculation lmax (for accuracy)
    lmax_output=3000,       # Output lmax
    delensing_efficiency=0.8, # Delensing efficiency [fraction]
    output_csv_path="data/result.csv"
):
    r"""
    Compute the delensed CMB temperature power spectrum l(l+1)C_l^{TT}/(2\pi) for a flat Lambda CDM cosmology
    using CAMB, applying a specified delensing efficiency, and save the results to a CSV file.

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
        Scalar amplitude of primordial perturbations.
    ns : float
        Scalar spectral index.
    lmax_calc : int
        Maximum multipole for internal calculation (should be >= lmax_output).
    lmax_output : int
        Maximum multipole for output (results will be for l=2 to lmax_output).
    delensing_efficiency : float
        Fraction of lensing potential power removed (0=no delensing, 1=perfect delensing).
    output_csv_path : str
        Path to save the output CSV file.

    Returns
    -------
    None
        Saves the results to a CSV file and prints a summary to the console.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set cosmological parameters
    pars = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu,
        omk=omk,
        tau=tau,
        As=As,
        ns=ns
    )
    # Set calculation lmax and lensing accuracy
    pars.set_for_lmax(lmax=lmax_calc, lens_potential_accuracy=1)

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Compute residual lensing amplitude (Alens)
    Alens_residual = 1.0 - delensing_efficiency

    # Get delensed CMB power spectra in muK^2, l(l+1)C_l/2pi
    delensed_cls = results.get_partially_lensed_cls(
        Alens=Alens_residual,
        lmax=lmax_output,
        CMB_unit='muK',
        raw_cl=False
    )
    # Extract multipole moments and TT spectrum
    ls = np.arange(2, lmax_output + 1)  # l=2 to lmax_output
    TT_delensed = delensed_cls[2:lmax_output + 1, 0]  # TT column, l=2 to lmax_output

    # Save to CSV
    df = pd.DataFrame({'l': ls, 'TT': TT_delensed})
    df.to_csv(output_csv_path, index=False)

    # Print summary
    print("Delensed CMB TT power spectrum computed and saved to " + output_csv_path)
    print("Cosmological parameters used:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  Sum m_nu = " + str(mnu) + " eV")
    print("  Omega_k = " + str(omk))
    print("  tau = " + str(tau))
    print("  As = " + str(As))
    print("  ns = " + str(ns))
    print("Delensing efficiency: " + str(delensing_efficiency * 100.0) + " % (Alens = " + str(Alens_residual) + ")")
    print("Output multipole range: l = 2 to " + str(lmax_output))
    print("First 5 rows of the output (l, TT [muK^2]):")
    print(df.head().to_string(index=False))
    print("Last 5 rows of the output (l, TT [muK^2]):")
    print(df.tail().to_string(index=False))


if __name__ == "__main__":
    compute_delensed_cmb_tt_spectrum()