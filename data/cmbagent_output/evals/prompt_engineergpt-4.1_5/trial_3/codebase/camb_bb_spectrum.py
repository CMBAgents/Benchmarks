# filename: codebase/camb_bb_spectrum.py
import os
import numpy as np
import pandas as pd

# Ensure data directory exists
 data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

try:
    import camb
    from camb import model, initialpower
except ImportError as e:
    print("CAMB is not installed. Please install camb and rerun the code.")
    raise e

def compute_cmb_bb_spectrum():
    r"""
    Compute the CMB B-mode polarization power spectrum (l(l+1)C_l^{BB}/(2pi)) for the specified cosmology.
    Returns:
        l_vals (np.ndarray): Multipole moments (l), shape (N,), units: dimensionless
        BB_vals (np.ndarray): B-mode power spectrum (l(l+1)C_l^{BB}/(2pi)), shape (N,), units: uK^2
    """
    # Set cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    r_tensor = 0.0  # Tensor-to-scalar ratio [dimensionless]
    As = 2e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]

    # Multipole range
    lmin = 2
    lmax = 3000

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=r_tensor)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.WantTensors = True  # Needed for BB spectrum, even if r=0

    # Run CAMB
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total'])

    # Extract BB spectrum
    cl = powers['total']  # shape: (lmax+1, 4) [TT, EE, BB, TE]
    # cl[:,2] is BB
    l_vals = np.arange(cl.shape[0])  # l = 0, 1, ..., lmax
    BB = cl[:,2]  # units: muK^2

    # Compute l(l+1)C_l^{BB}/(2pi) for l=2 to lmax
    l_range = np.arange(lmin, lmax+1)
    BB_cl = BB[lmin:lmax+1]  # Select l=2 to lmax
    BB_power = l_range * (l_range + 1) * BB_cl / (2.0 * np.pi)  # units: muK^2

    return l_range, BB_power

def save_results_to_csv(l_vals, BB_vals, filename):
    r"""
    Save the B-mode power spectrum to a CSV file.
    Args:
        l_vals (np.ndarray): Multipole moments (l), shape (N,)
        BB_vals (np.ndarray): B-mode power spectrum, shape (N,)
        filename (str): Output CSV filename
    """
    df = pd.DataFrame({'l': l_vals, 'BB': BB_vals})
    df.to_csv(filename, index=False)
    print("Saved B-mode power spectrum to " + filename)
    # Print a detailed summary of the results
    pd.set_option("display.precision", 8)
    pd.set_option("display.max_rows", 10)
    print("\nFirst and last 5 rows of the B-mode power spectrum (l, BB [uK^2]):")
    print(pd.concat([df.head(5), df.tail(5)]))

if __name__ == "__main__":
    l_vals, BB_vals = compute_cmb_bb_spectrum()
    output_file = os.path.join("data", "result.csv")
    save_results_to_csv(l_vals, BB_vals, output_file)