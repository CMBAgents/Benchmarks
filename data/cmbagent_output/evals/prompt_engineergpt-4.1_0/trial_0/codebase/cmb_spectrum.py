# filename: codebase/cmb_spectrum.py
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
    print("CAMB is not installed. Please install the camb package to run this code.")
    raise e


def compute_cmb_tt_spectrum():
    r"""
    Computes the CMB temperature power spectrum for the specified flat Lambda CDM cosmology.

    Returns:
        l_vals (np.ndarray): Multipole moments (l), shape (N,), unitless.
        tt_spectrum (np.ndarray): Temperature power spectrum l(l+1)C_l^{TT}/(2pi) in uK^2, shape (N,).
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.02  # Omega_b h^2
    omch2 = 0.122  # Omega_c h^2
    mnu = 0.06  # Sum of neutrino masses [eV]
    omk = 0.0  # Flat universe
    tau = 0.06  # Optical depth
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index

    # Multipole range
    lmin = 2
    lmax = 3000

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.WantTensors = False

    # Calculate results for these parameters
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    cl = powers['total']  # shape: (lmax+1, 4), columns: TT, EE, BB, TE

    # l values: cl[ell, ...] for ell = 0, 1, ..., lmax
    l_vals = np.arange(lmin, lmax + 1)
    # TT spectrum: cl[:,0] is C_l^{TT} in (uK)^2
    cl_tt = cl[lmin:lmax+1, 0]  # (lmax-lmin+1,)

    # Compute l(l+1)C_l^{TT}/(2pi) in uK^2
    tt_spectrum = l_vals * (l_vals + 1) * cl_tt / (2.0 * np.pi)

    return l_vals, tt_spectrum


def save_spectrum_to_csv(l_vals, tt_spectrum, filename):
    r"""
    Save the CMB TT power spectrum to a CSV file.

    Args:
        l_vals (np.ndarray): Multipole moments (l), shape (N,).
        tt_spectrum (np.ndarray): TT power spectrum, shape (N,).
        filename (str): Output CSV file path.
    """
    df = pd.DataFrame({'l': l_vals, 'TT': tt_spectrum})
    df.to_csv(filename, index=False)
    print("Saved CMB TT power spectrum to " + filename)
    print("First 10 rows:")
    print(df.head(10).to_string(index=False))
    print("\nLast 10 rows:")
    print(df.tail(10).to_string(index=False))
    print("\nTotal number of multipoles: " + str(len(df)))


if __name__ == "__main__":
    l_vals, tt_spectrum = compute_cmb_tt_spectrum()
    output_csv = os.path.join("data", "result.csv")
    save_spectrum_to_csv(l_vals, tt_spectrum, output_csv)
