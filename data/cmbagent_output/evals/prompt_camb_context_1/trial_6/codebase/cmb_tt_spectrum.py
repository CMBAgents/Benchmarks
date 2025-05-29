# filename: codebase/cmb_tt_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_tt_spectrum(
    H0=67.3,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density Omega_b h^2
    omch2=0.122,            # Cold dark matter density Omega_c h^2
    mnu=0.06,               # Neutrino mass sum [eV]
    omk=0.05,               # Curvature Omega_k
    tau=0.06,               # Optical depth to reionization
    As=2.0e-9,              # Scalar amplitude
    ns=0.965,               # Scalar spectral index
    lmax=3000,              # Maximum multipole
    output_dir="data/",     # Output directory
    output_filename="result.csv" # Output CSV filename
):
    r"""
    Compute the lensed CMB temperature power spectrum D_l^{TT} = l(l+1)C_l^{TT}/(2\pi) in μK^2
    for a non-flat Lambda CDM cosmology using CAMB, for multipoles l=2 to lmax.

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
    lmax : int
        Maximum multipole to compute.
    output_dir : str
        Directory to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
        Saves the computed spectrum to a CSV file and prints a summary.
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
        WantScalars=True,
        WantTensors=False
    )

    # Run CAMB and get results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in μK^2
    powers = results.get_cmb_power_spectra(params=pars, CMB_unit='muK', spectra=['lensed_scalar'])
    lensed_cls = powers['lensed_scalar']  # shape: (lmax+1, 4)
    tt_spectrum_Dl = lensed_cls[:, 0]     # TT column

    # Prepare data for l=2 to lmax
    l_values = np.arange(2, lmax + 1)
    tt_values = tt_spectrum_Dl[2:lmax + 1]

    # Save to CSV
    output_path = os.path.join(output_dir, output_filename)
    df = pd.DataFrame({'l': l_values, 'TT': tt_values})
    df.to_csv(output_path, index=False)

    # Print summary
    print("CMB lensed TT power spectrum (Dl = l(l+1)Cl/2pi) in μK^2 saved to " + output_path)
    print("First 5 rows of the output:")
    print(df.head())
    print("Last 5 rows of the output:")
    print(df.tail())
    print("Total multipoles saved: " + str(len(df)))
    print("Parameter summary:")
    print("H0 = " + str(H0) + " km/s/Mpc, ombh2 = " + str(ombh2) + ", omch2 = " + str(omch2) + ", mnu = " + str(mnu) + " eV, omk = " + str(omk) + ", tau = " + str(tau) + ", As = " + str(As) + ", ns = " + str(ns) + ", lmax = " + str(lmax))


if __name__ == "__main__":
    compute_cmb_tt_spectrum()
