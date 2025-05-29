# filename: codebase/compute_delensed_cmb_tt_spectrum.py
import camb
import numpy as np
import os
import pandas as pd

def compute_delensed_cmb_tt_spectrum(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density Omega_b h^2
    omch2=0.122,            # Cold dark matter density Omega_c h^2
    mnu=0.06,               # Neutrino mass sum [eV]
    omk=0.0,                # Curvature Omega_k
    tau=0.06,               # Optical depth to reionization
    As=2.0e-9,              # Scalar amplitude
    ns=0.965,               # Scalar spectral index
    delensing_efficiency=0.8, # Fraction of lensing removed (80%)
    lmin=2,                 # Minimum multipole
    lmax=3000,              # Maximum multipole for output
    lmax_calc=3500,         # Maximum multipole for internal calculation
    output_csv='data/result.csv' # Output CSV file path
):
    r"""
    Compute the delensed CMB temperature power spectrum D_l^{TT} = l(l+1)C_l^{TT}/(2pi)
    for a flat Lambda CDM cosmology using CAMB, applying a specified delensing efficiency.

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
        Scalar amplitude of primordial power spectrum.
    ns : float
        Scalar spectral index.
    delensing_efficiency : float
        Fraction of lensing potential power removed (0 to 1).
    lmin : int
        Minimum multipole to include in output.
    lmax : int
        Maximum multipole to include in output.
    lmax_calc : int
        Maximum multipole for internal calculation (should be >= lmax).
    output_csv : str
        Path to output CSV file.

    Returns
    -------
    None
        Saves the delensed TT power spectrum to a CSV file and prints a summary.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax=lmax_calc, lens_potential_accuracy=1)
    pars.WantScalars = True
    pars.DoLensing = True
    pars.WantTensors = False

    # Run CAMB
    results = camb.get_results(pars)

    # Compute delensed spectra: Alens = 1 - delensing_efficiency
    Alens_param = 1.0 - delensing_efficiency
    delensed_cls = results.get_partially_lensed_cls(
        Alens=Alens_param,
        lmax=lmax,
        CMB_unit='muK',  # Output in muK^2
        raw_cl=False     # Output D_l = l(l+1)C_l/(2pi)
    )

    # Extract l and TT spectrum
    ls = np.arange(lmin, lmax + 1)
    TT = delensed_cls[lmin:lmax+1, 0]  # D_l^{TT} in muK^2

    # Save to CSV
    df = pd.DataFrame({'l': ls, 'TT': TT})
    df.to_csv(output_csv, index=False)

    # Print summary
    print("Delensed CMB TT power spectrum (D_l = l(l+1)C_l^{TT}/(2pi)) saved to " + output_csv)
    print("Units: D_l in microkelvin^2 (muK^2), l is dimensionless.")
    print("Cosmological parameters:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  Sum m_nu = " + str(mnu) + " eV")
    print("  Omega_k = " + str(omk))
    print("  tau = " + str(tau))
    print("  A_s = " + str(As))
    print("  n_s = " + str(ns))
    print("Delensing efficiency: " + str(delensing_efficiency * 100.0) + " % (Alens = " + str(Alens_param) + ")")
    print("l range: " + str(lmin) + " to " + str(lmax))
    print("First 5 rows of the output:")
    print(df.head(5).to_string(index=False))
    print("Last 5 rows of the output:")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    compute_delensed_cmb_tt_spectrum()