# filename: codebase/camb_cmb_tt_spectrum.py
import os
import numpy as np
import pandas as pd

# Ensure output directory exists
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    import camb
    from camb import model, initialpower
except ImportError as e:
    print("CAMB package is required but not installed. Please install camb and rerun the code.")
    raise e

def compute_cmb_tt_spectrum(
    H0=70.0,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density Omega_b h^2
    omch2=0.122,            # Cold dark matter density Omega_c h^2
    mnu=0.06,               # Neutrino mass sum [eV]
    omk=0.0,                # Curvature Omega_k
    tau=0.06,               # Optical depth to reionization
    As=2e-9,                # Scalar amplitude
    ns=0.965,               # Scalar spectral index
    lmin=2,                 # Minimum multipole
    lmax=3000,              # Maximum multipole
    output_csv="data/result.csv"
):
    r"""
    Compute the CMB temperature power spectrum (C_l^{TT}) for a flat Lambda CDM cosmology.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter Omega_b h^2.
    omch2 : float
        Physical cold dark matter density parameter Omega_c h^2.
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
    lmin : int
        Minimum multipole moment (inclusive).
    lmax : int
        Maximum multipole moment (inclusive).
    output_csv : str
        Path to output CSV file.

    Returns
    -------
    None
        Saves the results to a CSV file with columns 'l' and 'TT' (in microkelvin^2).
    """
    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.WantTensors = False

    # Calculate results for these parameters
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    # 'total' contains the raw Cls (unlensed, no foregrounds)
    cl = powers['total']

    # cl shape: (lmax+1, 4) columns: TT, EE, BB, TE
    # l=0,1 are not physical, so skip them
    ell = np.arange(cl.shape[0])
    mask = (ell >= lmin) & (ell <= lmax)
    ell = ell[mask]
    cl_tt = cl[mask, 0]  # TT spectrum in muK^2

    # Save to CSV
    df = pd.DataFrame({'l': ell, 'TT': cl_tt})
    df.to_csv(output_csv, index=False)

    # Print summary
    pd.set_option("display.precision", 8)
    pd.set_option("display.max_rows", 10)
    print("CMB TT power spectrum (C_l^{TT}) computed for l = " + str(lmin) + " to " + str(lmax) + ".")
    print("Results saved to " + output_csv)
    print("First and last 5 rows of the result:")
    print(pd.concat([df.head(5), df.tail(5)]))

if __name__ == "__main__":
    compute_cmb_tt_spectrum()