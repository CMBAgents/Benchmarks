# filename: codebase/compute_delensed_cmb_tt_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_delensed_cmb_tt_spectrum(
    H0=67.5,
    ombh2=0.022,
    omch2=0.122,
    mnu=0.06,
    omk=0.0,
    tau=0.06,
    As=2.0e-9,
    ns=0.965,
    lmax=3000,
    lens_potential_accuracy=1,
    delensing_efficiency=0.8,
    output_folder="data/",
    output_filename="result.csv"
):
    r"""
    Compute the delensed CMB temperature power spectrum l(l+1)C_l^{TT}/(2pi) in muK^2
    for a flat Lambda CDM cosmology using CAMB, with specified cosmological parameters and
    delensing efficiency. Save the result as a CSV file with columns 'l' and 'TT'.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter (\Omega_b h^2).
    omch2 : float
        Physical cold dark matter density parameter (\Omega_c h^2).
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter (\Omega_k).
    tau : float
        Optical depth to reionization.
    As : float
        Scalar amplitude of primordial power spectrum.
    ns : float
        Scalar spectral index.
    lmax : int
        Maximum multipole moment to compute.
    lens_potential_accuracy : int
        Accuracy setting for lensing potential calculation (1 is Planck-level).
    delensing_efficiency : float
        Fraction of lensing power removed (e.g., 0.8 for 80 percent delensing).
    output_folder : str
        Directory to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
        The function saves the result to a CSV file and prints a summary to the console.
    """
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Set cosmological parameters
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
        lens_potential_accuracy=lens_potential_accuracy
    )

    # Compute results
    results = camb.get_results(pars)

    # Compute delensed CMB power spectra
    # Delensing efficiency: residual lensing power is (1 - delensing_efficiency)
    Alens_residual = 1.0 - delensing_efficiency

    # Get lensed spectra with reduced lensing potential
    # Output: CL[0:lmax+1, 0:4] (TT, EE, BB, TE), in muK^2, Dl = l(l+1)Cl/2pi
    delensed_cls = results.get_partially_lensed_cls(
        Alens=Alens_residual,
        lmax=lmax,
        CMB_unit='muK',
        raw_cl=False
    )

    # Extract l and TT spectrum for l=2 to l=lmax
    ls = np.arange(2, lmax + 1)
    tt = delensed_cls[2:lmax + 1, 0]

    # Save to CSV
    output_path = os.path.join(output_folder, output_filename)
    df = pd.DataFrame({'l': ls.astype(int), 'TT': tt})
    df.to_csv(output_path, index=False)

    # Print summary
    print("Delensed CMB TT power spectrum (l(l+1)C_l^{TT}/(2pi) in muK^2) saved to " + output_path)
    print("First five rows:")
    print(df.head())
    print("Last five rows:")
    print(df.tail())
    print("Total number of multipoles: " + str(len(df)))
    print("Cosmological parameters used:")
    print("H0 = " + str(H0) + " km/s/Mpc, ombh2 = " + str(ombh2) + ", omch2 = " + str(omch2) + ", mnu = " + str(mnu) + " eV, omk = " + str(omk) + ", tau = " + str(tau) + ", As = " + str(As) + ", ns = " + str(ns))
    print("Delensing efficiency: " + str(delensing_efficiency * 100) + " percent")


if __name__ == "__main__":
    compute_delensed_cmb_tt_spectrum()