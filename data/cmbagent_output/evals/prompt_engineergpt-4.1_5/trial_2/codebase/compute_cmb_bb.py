# filename: codebase/compute_cmb_bb.py
import os
import numpy as np
import pandas as pd

# Ensure data directory exists
database_path = "data"
if not os.path.exists(database_path):
    os.makedirs(database_path)

try:
    import camb
    from camb import model, initialpower
except ImportError as e:
    print("CAMB is not installed. Please install camb and rerun the code.")
    raise e

def compute_cmb_bb_spectrum():
    """
    Computes the CMB B-mode polarization power spectrum (l(l+1)C_l^{BB}/(2pi)) for the specified cosmology.
    Returns:
        l_vals (np.ndarray): Multipole moments (l), shape (N,), unitless (integer)
        BB_muK2 (np.ndarray): B-mode power spectrum (l(l+1)C_l^{BB}/(2pi)), shape (N,), units: micro-Kelvin^2
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density Omega_b h^2
    omch2 = 0.122  # Cold dark matter density Omega_c h^2
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature Omega_k
    tau = 0.06  # Optical depth to reionization
    r_tensor = 0.0  # Tensor-to-scalar ratio
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index

    # Multipole range
    lmin = 2
    lmax = 3000

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=r_tensor)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.WantTensors = True  # Needed for BB spectrum, even if r=0

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total'])

    # Extract BB spectrum
    cl = powers['total']  # shape: (lmax+1, 4) for TT, EE, BB, TE
    # cl[:,2] is BB in muK^2 units
    l_vals = np.arange(cl.shape[0])  # l = 0, 1, ..., lmax
    BB = cl[:,2]  # BB spectrum, units: muK^2

    # Compute l(l+1)C_l^{BB}/(2pi)
    factor = l_vals * (l_vals + 1) / (2.0 * np.pi)
    BB_muK2 = factor * BB  # units: muK^2

    # Restrict to l=2 to l=3000
    mask = (l_vals >= lmin) & (l_vals <= lmax)
    l_vals = l_vals[mask]
    BB_muK2 = BB_muK2[mask]

    return l_vals, BB_muK2

def save_results_to_csv(l_vals, BB_muK2, filename):
    """
    Save the B-mode power spectrum results to a CSV file.
    Args:
        l_vals (np.ndarray): Multipole moments (l), shape (N,)
        BB_muK2 (np.ndarray): B-mode power spectrum, shape (N,), units: micro-Kelvin^2
        filename (str): Output CSV filename
    """
    df = pd.DataFrame({'l': l_vals, 'BB': BB_muK2})
    df.to_csv(filename, index=False)
    print("Saved B-mode power spectrum to " + filename)
    print("First 10 rows:")
    print(df.head(10).to_string(index=False))
    print("Last 10 rows:")
    print(df.tail(10).to_string(index=False))
    print("Total number of multipoles: " + str(len(df)))

if __name__ == "__main__":
    l_vals, BB_muK2 = compute_cmb_bb_spectrum()
    output_csv = os.path.join(database_path, "result.csv")
    save_results_to_csv(l_vals, BB_muK2, output_csv)