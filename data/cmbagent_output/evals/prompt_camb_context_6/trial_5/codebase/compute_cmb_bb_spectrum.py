# filename: codebase/compute_cmb_bb_spectrum.py
import camb
import numpy as np
import pandas as pd
import os


def compute_cmb_bb_spectrum():
    r"""
    Compute the Cosmic Microwave Background (CMB) B-mode polarization power spectrum (C_l^{BB})
    for a flat Lambda CDM cosmology using CAMB, with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Neutrino mass sum (mnu): 0.06 eV
        - Curvature (omk): 0
        - Optical depth to reionization (tau): 0.06
        - Tensor-to-scalar ratio (r): 0.1
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965

    The function computes the B-mode power spectrum (C_l^{BB}) in units of microkelvin squared (\u03bcK^2)
    for multipole moments l=2 to l=3000, and saves the results in a CSV file 'data/result.csv'
    with columns:
        - l: Multipole moment (integer, 2 to 3000)
        - BB: B-mode polarization power spectrum (C_l^{BB} in \u03bcK^2)
    """
    # Cosmological parameters (units in comments)
    H0 = 67.5                # Hubble constant [km/s/Mpc]
    ombh2 = 0.022            # Omega_b h^2 [dimensionless]
    omch2 = 0.122            # Omega_c h^2 [dimensionless]
    mnu = 0.06               # Sum of neutrino masses [eV]
    omk = 0.0                # Curvature [dimensionless]
    tau = 0.06               # Optical depth [dimensionless]
    r_tensor = 0.1           # Tensor-to-scalar ratio [dimensionless]
    As = 2e-9                # Scalar amplitude [dimensionless]
    ns = 0.965               # Scalar spectral index [dimensionless]
    lmax = 3000              # Maximum multipole moment

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=r_tensor)
    pars.WantTensors = True
    pars.DoLensing = True
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)

    # Run CAMB and get results
    results = camb.get_results(pars)

    # Get total (lensed scalar + tensor) CMB power spectra in \u03bcK^2, raw Cl
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True, spectra=['total'])
    cl_bb = powers['total'][:, 2]  # BB is column 2

    # Prepare l and BB arrays for l=2..3000
    l_arr = np.arange(2, lmax + 1)
    bb_arr = cl_bb[2:lmax + 1]

    # Prepare output directory and filename
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_csv = os.path.join(output_dir, "result.csv")

    # Save to CSV
    df = pd.DataFrame({'l': l_arr, 'BB': bb_arr})
    df.to_csv(output_csv, index=False)

    # Print detailed summary
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print("CMB B-mode (BB) power spectrum computed for flat Lambda CDM cosmology with:")
    print("  H0 = " + str(H0) + " km/s/Mpc, ombh2 = " + str(ombh2) + ", omch2 = " + str(omch2) + ", mnu = " + str(mnu) + " eV")
    print("  omk = " + str(omk) + ", tau = " + str(tau) + ", r = " + str(r_tensor) + ", As = " + str(As) + ", ns = " + str(ns))
    print("Multipole range: l = 2 to " + str(lmax))
    print("Results saved to: " + output_csv)
    print("First 5 rows:")
    print(df.head(5).to_string(index=False))
    print("Last 5 rows:")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    compute_cmb_bb_spectrum()
