# filename: codebase/compute_cmb_bb_spectrum.py
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower
import os


def compute_cmb_bb_spectrum():
    r"""
    Compute the CMB B-mode polarization power spectrum (C_l^{BB}) for a flat Lambda CDM cosmology
    using CAMB, with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (Omega_b h^2): 0.022
        - Cold dark matter density (Omega_c h^2): 0.122
        - Neutrino mass sum (sum m_nu): 0.06 eV
        - Curvature (Omega_k): 0
        - Optical depth to reionization (tau): 0.06
        - Tensor-to-scalar ratio (r): 0.1
        - Scalar amplitude (A_s): 2e-9
        - Scalar spectral index (n_s): 0.965

    Computes the B-mode power spectrum (C_l^{BB}) in units of micro-Kelvin^2 for multipole moments
    from l=2 to l=3000. Saves the results in a CSV file named 'result.csv' with columns:
        - l: Multipole moment (integer values from 2 to 3000)
        - BB: B-mode polarization power spectrum (C_l^{BB} in micro-Kelvin^2)
    """
    # Output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "result.csv")

    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    r_tensor = 0.1  # Tensor-to-scalar ratio [dimensionless]
    As = 2e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]

    # Multipole range
    lmin = 2
    lmax = 3000

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=r_tensor)
    pars.WantTensors = True
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.Want_CMB = True
    pars.Want_CMB_lensing = False
    pars.Want_cl_2D_array = True

    # Compute results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
    # powers['total'] has shape (lmax+1, 4): columns are TT, EE, BB, TE

    # Extract BB spectrum
    cl_total = powers['total']  # shape: (lmax+1, 4)
    # cl_total[l, 2] is BB at multipole l
    ells = np.arange(cl_total.shape[0])  # l = 0, 1, ..., lmax
    BB = cl_total[:, 2]  # BB in muK^2

    # Select l=2 to l=3000
    mask = (ells >= lmin) & (ells <= lmax)
    l_vals = ells[mask]
    BB_vals = BB[mask]

    # Save to CSV
    df = pd.DataFrame({'l': l_vals, 'BB': BB_vals})
    df.to_csv(output_file, index=False)

    # Print summary
    pd.set_option("display.precision", 8)
    pd.set_option("display.max_rows", 10)
    print("CMB B-mode polarization power spectrum (C_l^{BB}) computed for l=2 to l=3000.")
    print("Units: l (dimensionless), BB (micro-Kelvin^2)")
    print("First and last 5 rows of the result:")
    print(pd.concat([df.head(5), df.tail(5)]))
    print("\nResults saved to " + output_file)


if __name__ == "__main__":
    compute_cmb_bb_spectrum()