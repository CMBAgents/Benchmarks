# filename: codebase/cmb_bb_spectrum.py
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower
import os

def compute_cmb_bb_spectrum():
    r"""
    Compute the CMB B-mode polarization power spectrum (C_l^{BB}) for a flat Lambda CDM cosmology
    using CAMB, with the following parameters:
        - Hubble constant (H_0): 67.5 km/s/Mpc
        - Baryon density (Omega_b h^2): 0.022
        - Cold dark matter density (Omega_c h^2): 0.122
        - Neutrino mass sum (Sigma m_nu): 0.06 eV
        - Curvature (Omega_k): 0
        - Optical depth to reionization (tau): 0.06
        - Tensor-to-scalar ratio (r): 0.1
        - Scalar amplitude (A_s): 2e-9
        - Scalar spectral index (n_s): 0.965

    Computes the B-mode power spectrum (C_l^{BB}) in units of micro-Kelvin^2 for multipole moments
    from l=2 to l=3000. Saves the results in a CSV file named 'result.csv' with two columns:
        - l: Multipole moment (integer values from 2 to 3000)
        - BB: B-mode polarization power spectrum (C_l^{BB} in micro-Kelvin^2)
    """
    # Output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_csv = os.path.join(output_dir, "result.csv")

    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density
    omch2 = 0.122  # Cold dark matter density
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature
    tau = 0.06  # Optical depth to reionization
    r_tensor = 0.1  # Tensor-to-scalar ratio
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index

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

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    # 'total' includes lensing, 'unlensed_scalar' is scalar only, 'tensor' is tensor only, etc.
    # For raw BB, use 'total' (includes both scalar and tensor, but BB is only from tensor at this r)
    cl = powers['total']  # shape: (lmax+1, 4): columns are TT, EE, BB, TE

    # Extract BB spectrum (column 2)
    BB = cl[lmin:lmax+1, 2]  # BB in muK^2
    ell = np.arange(lmin, lmax+1)

    # Save to CSV
    df = pd.DataFrame({'l': ell, 'BB': BB})
    df.to_csv(output_csv, index=False)

    # Print summary
    pd.set_option("display.precision", 8)
    pd.set_option("display.max_rows", 10)
    print("CMB B-mode polarization power spectrum (C_l^{BB}) computed for l = 2 to 3000.")
    print("Units: BB in micro-Kelvin^2 (muK^2).")
    print("First and last 5 rows of the result:")
    print(pd.concat([df.head(5), df.tail(5)]))
    print("\nResults saved to " + output_csv)


if __name__ == "__main__":
    compute_cmb_bb_spectrum()
