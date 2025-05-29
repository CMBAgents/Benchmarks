# filename: codebase/compute_cmb_ee_spectrum.py
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
    print("CAMB is required for this code. Please install camb and rerun.")
    raise e

def compute_cmb_ee_spectrum():
    r"""
    Computes the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for the specified cosmology.
    Returns:
        l_vals (np.ndarray): Multipole moments (l), shape (N,), units: dimensionless
        cl_ee (np.ndarray): E-mode power spectrum (microkelvin^2), shape (N,), units: microkelvin^2
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

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(3000, lens_potential_accuracy=0)
    pars.WantTensors = False

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=3000)
    # 'total' includes l=0,1, so index 2 corresponds to l=2
    cl = powers['total']  # shape: (lmax+1, 4) [TT, EE, BB, TE]

    # Extract l and EE spectrum
    lmin = 2
    lmax = 3000
    l_vals = np.arange(lmin, lmax + 1)  # l=2..3000
    # cl[:,1] is EE, cl has shape (lmax+1, 4)
    cl_ee = cl[lmin:lmax+1, 1]  # EE, units: microkelvin^2

    # Convert to l(l+1)C_l/(2pi)
    factor = l_vals * (l_vals + 1) / (2.0 * np.pi)
    cl_ee = factor * cl_ee  # units: microkelvin^2

    return l_vals, cl_ee

def save_ee_spectrum_to_csv(l_vals, cl_ee, filename):
    r"""
    Saves the E-mode power spectrum to a CSV file.
    Args:
        l_vals (np.ndarray): Multipole moments (l), shape (N,)
        cl_ee (np.ndarray): E-mode power spectrum (microkelvin^2), shape (N,)
        filename (str): Output CSV filename
    """
    df = pd.DataFrame({'l': l_vals.astype(int), 'EE': cl_ee})
    df.to_csv(filename, index=False)
    print("Saved E-mode polarization power spectrum to " + filename)
    print("First 10 rows:")
    print(df.head(10).to_string(index=False))
    print("Last 10 rows:")
    print(df.tail(10).to_string(index=False))
    print("Total rows: " + str(len(df)))

if __name__ == "__main__":
    l_vals, cl_ee = compute_cmb_ee_spectrum()
    output_file = os.path.join("data", "result.csv")
    save_ee_spectrum_to_csv(l_vals, cl_ee, output_file)
