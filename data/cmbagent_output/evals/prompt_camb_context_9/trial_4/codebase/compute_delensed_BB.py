# filename: codebase/compute_delensed_BB.py
import camb
import numpy as np
import pandas as pd
import os

def compute_delensed_BB_spectrum():
    r"""
    Compute the raw delensed CMB B-mode polarization power spectrum (C_ell^{BB}) for a flat Lambda CDM cosmology
    using CAMB, with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Neutrino mass sum (mnu): 0.06 eV
        - Curvature (omk): 0
        - Optical depth to reionization (tau): 0.06
        - Tensor-to-scalar ratio (r): 0.1
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965
    The calculation includes tensor modes and applies a delensing efficiency of 10% (i.e., 10% of lensing B-mode
    power is removed). The output is the total delensed B-mode power spectrum (scalar + tensor) in units of muK^2,
    for multipole moments l=2 to l=3000, saved as a CSV file with columns:
        - l: Multipole moment (integer)
        - BB: Delensed B-mode power spectrum (muK^2)
    The file is saved as 'data/result.csv'.
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [Omega_b h^2]
    omch2 = 0.122  # Cold dark matter density [Omega_c h^2]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    r_tensor = 0.1  # Tensor-to-scalar ratio [dimensionless]
    As_scalar = 2e-9  # Scalar amplitude [dimensionless]
    ns_scalar = 0.965  # Scalar spectral index [dimensionless]

    # Delensing efficiency: 10% (i.e., 10% of lensing B-mode power is removed)
    delensing_efficiency = 0.1  # [dimensionless]
    Alens_param = 1.0 - delensing_efficiency  # Scaling factor for lensing potential

    l_max_calc = 3000  # Maximum multipole moment

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As_scalar, ns=ns_scalar, r=r_tensor)
    pars.set_for_lmax(lmax=l_max_calc, lens_potential_accuracy=1)
    pars.WantTensors = True

    # Run CAMB
    results = camb.get_results(pars)

    # Get partially delensed scalar BB spectrum (index 2)
    cl_bb_scalar_delensed = results.get_partially_lensed_cls(
        Alens=Alens_param,
        lmax=l_max_calc,
        CMB_unit='muK',
        raw_cl=True
    )[:, 2]  # [muK^2]

    # Get tensor BB spectrum (index 2)
    cl_bb_tensor = results.get_tensor_cls(
        lmax=l_max_calc,
        CMB_unit='muK',
        raw_cl=True
    )[:, 2]  # [muK^2]

    # Total delensed BB = (delensed scalar BB) + (tensor BB)
    cl_bb_total_delensed = cl_bb_scalar_delensed + cl_bb_tensor  # [muK^2]

    # Prepare output for l=2 to 3000
    ls = np.arange(2, l_max_calc + 1)  # [dimensionless]
    bb_output = cl_bb_total_delensed[2:l_max_calc + 1]  # [muK^2]

    # Prepare output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'l': ls, 'BB': bb_output})
    df.to_csv(output_path, index=False, float_format='%.8e')

    # Print summary to console
    print("Delensed CMB B-mode power spectrum (C_ell^{BB}) computed and saved to " + output_path)
    print("Columns:")
    print("  l  : Multipole moment (integer, 2 to 3000)")
    print("  BB : Delensed B-mode power spectrum (muK^2)")
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())


if __name__ == "__main__":
    compute_delensed_BB_spectrum()