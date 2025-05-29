# filename: codebase/camb_cmb_tt_spectrum.py
import os
import numpy as np
import pandas as pd

# Ensure data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

try:
    import camb
    from camb import model, initialpower
except ImportError as e:
    print("CAMB package is required but not installed. Please install camb and rerun the code.")
    raise e


def compute_cmb_tt_spectrum(
    H0=70.0,           # Hubble constant [km/s/Mpc]
    ombh2=0.022,       # Omega_b h^2
    omch2=0.122,       # Omega_c h^2
    mnu=0.06,          # Sum of neutrino masses [eV]
    omk=0.0,           # Omega_k (curvature)
    tau=0.06,          # Optical depth to reionization
    As=2e-9,           # Scalar amplitude
    ns=0.965,          # Scalar spectral index
    lmax=3000          # Maximum multipole
):
    r"""
    Compute the CMB temperature power spectrum (C_l^{TT}) using CAMB.

    Returns:
        l (ndarray): Multipole moments (l=2 to lmax)
        cl_tt (ndarray): Temperature power spectrum (C_l^{TT}) in microKelvin^2
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
    cl = powers['unlensed_scalar']  # shape: (lmax+1, 4): TT, EE, BB, TE

    # l values start from 0, but CMB spectrum is meaningful for l >= 2
    l = np.arange(cl.shape[0])
    lmin = 2
    lmax = min(lmax, cl.shape[0] - 1)
    l = l[lmin:lmax+1]
    cl_tt = cl[lmin:lmax+1, 0]  # TT spectrum in muK^2

    return l, cl_tt


def save_spectrum_to_csv(l, cl_tt, filename):
    r"""
    Save the multipole moments and TT power spectrum to a CSV file.

    Args:
        l (ndarray): Multipole moments.
        cl_tt (ndarray): TT power spectrum in microKelvin^2.
        filename (str): Output CSV file path.
    """
    df = pd.DataFrame({'l': l, 'TT': cl_tt})
    df.to_csv(filename, index=False)
    print("Saved CMB TT power spectrum to " + filename)
    print("First 10 rows:")
    print(df.head(10).to_string(index=False))
    print("Last 10 rows:")
    print(df.tail(10).to_string(index=False))
    print("Total number of multipoles: " + str(len(df)))


if __name__ == "__main__":
    l, cl_tt = compute_cmb_tt_spectrum(
        H0=70.0,
        ombh2=0.022,
        omch2=0.122,
        mnu=0.06,
        omk=0.0,
        tau=0.06,
        As=2e-9,
        ns=0.965,
        lmax=3000
    )
    save_spectrum_to_csv(l, cl_tt, "data/result.csv")
