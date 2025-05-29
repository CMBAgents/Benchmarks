# filename: codebase/compute_cmb_tt_spectrum.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

def compute_cmb_tt_spectrum(
    H0=74.0,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density [dimensionless]
    omch2=0.122,            # Cold dark matter density [dimensionless]
    mnu=0.06,               # Neutrino mass sum [eV]
    omk=0.0,                # Curvature [dimensionless]
    tau=0.06,               # Optical depth to reionization [dimensionless]
    As=2e-9,                # Scalar amplitude [dimensionless]
    ns=0.965,               # Scalar spectral index [dimensionless]
    lmin=2,                 # Minimum multipole
    lmax=3000,              # Maximum multipole
    output_csv="data/result.csv"
):
    r"""
    Compute the CMB temperature power spectrum (C_l^{TT}) for a flat Lambda CDM cosmology using CAMB.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter (Ω_b h^2).
    omch2 : float
        Physical cold dark matter density parameter (Ω_c h^2).
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter (Ω_k).
    tau : float
        Optical depth to reionization.
    As : float
        Scalar amplitude (A_s).
    ns : float
        Scalar spectral index (n_s).
    lmin : int
        Minimum multipole moment (l).
    lmax : int
        Maximum multipole moment (l).
    output_csv : str
        Path to output CSV file.

    Returns
    -------
    None
        Saves the results to a CSV file and prints a summary to the console.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.WantTensors = False

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
    cl = powers['unlensed_scalar']

    # Extract l and TT
    # cl shape: (lmax+1, 4), columns: TT, EE, BB, TE
    # l=0,1 are not physical, so skip to l=2
    l_vals = np.arange(cl.shape[0])
    l_mask = (l_vals >= lmin) & (l_vals <= lmax)
    l = l_vals[l_mask]
    TT = cl[l_mask, 0]  # TT in muK^2

    # Save to CSV
    df = pd.DataFrame({'l': l, 'TT': TT})
    df.to_csv(output_csv, index=False)

    # Print summary
    np.set_printoptions(precision=6, suppress=True)
    pd.set_option("display.precision", 6)
    pd.set_option("display.max_rows", 10)
    print("CMB TT power spectrum (C_l^{TT}) computed for l = " + str(lmin) + " to " + str(lmax) + ".")
    print("Results saved to " + output_csv)
    print("First and last 5 rows of the result:")
    print(pd.concat([df.head(5), df.tail(5)]))


if __name__ == "__main__":
    compute_cmb_tt_spectrum()