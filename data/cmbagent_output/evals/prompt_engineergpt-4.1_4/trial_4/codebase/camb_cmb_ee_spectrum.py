# filename: codebase/camb_cmb_ee_spectrum.py
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
    print("CAMB is not installed. Please install camb and rerun the code.")
    raise e


def compute_cmb_ee_spectrum():
    r"""
    Computes the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for the specified cosmology.
    Returns:
        l_vals (np.ndarray): Multipole moments (l), shape (N,), unitless
        cl_ee (np.ndarray): E-mode power spectrum (microkelvin^2), shape (N,)
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.04  # Optical depth to reionization [dimensionless]
    As = 2e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]

    # Multipole range
    lmax = 3000

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.WantTensors = False

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total'])

    # Extract EE spectrum
    totCL = powers['total']  # shape: (lmax+1, 4) [TT, EE, BB, TE]
    # l values: 0 to lmax
    l_vals = np.arange(totCL.shape[0])  # 0 to lmax
    # EE: column 1
    cl_ee = totCL[:, 1]  # units: microkelvin^2

    # Compute l(l+1)C_l^{EE}/(2pi)
    # Avoid l=0,1 (not defined)
    lmin = 2
    lmax_eff = min(lmax, len(l_vals)-1)
    l_vals = l_vals[lmin:lmax_eff+1]
    cl_ee = cl_ee[lmin:lmax_eff+1]
    cl_ee_scaled = l_vals * (l_vals + 1) * cl_ee / (2.0 * np.pi)  # units: microkelvin^2

    return l_vals, cl_ee_scaled


def save_to_csv(l_vals, cl_ee, filename):
    r"""
    Save the multipole moments and E-mode power spectrum to a CSV file.
    Args:
        l_vals (np.ndarray): Multipole moments (l)
        cl_ee (np.ndarray): E-mode power spectrum (microkelvin^2)
        filename (str): Output CSV filename
    """
    df = pd.DataFrame({'l': l_vals.astype(int), 'EE': cl_ee})
    df.to_csv(filename, index=False)
    print("Saved E-mode power spectrum to " + filename)
    # Print summary
    pd.set_option("display.precision", 6)
    pd.set_option("display.max_rows", 10)
    print("First and last 5 rows of the result:")
    print(pd.concat([df.head(5), df.tail(5)]))


if __name__ == "__main__":
    l_vals, cl_ee = compute_cmb_ee_spectrum()
    save_to_csv(l_vals, cl_ee, "data/result.csv")
