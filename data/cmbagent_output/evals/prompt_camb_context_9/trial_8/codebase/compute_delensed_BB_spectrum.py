# filename: codebase/compute_delensed_BB_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_delensed_BB_spectrum():
    r"""
    Compute the raw delensed CMB B-mode polarization power spectrum (C_ell^{BB}) for a flat Lambda CDM cosmology
    with tensors, using specified cosmological parameters and a 10% delensing efficiency.
    The result is saved as a CSV file with columns:
        l: Multipole moment (integer, 2 to 3000)
        BB: Delensed B-mode power spectrum (C_ell^{BB} in micro-Kelvin^2)
    The file is saved as 'data/result.csv'.

    Units:
        - All C_ell^{BB} values are in micro-Kelvin^2 (muK^2).
        - Multipole l is dimensionless.

    Returns
    -------
    None
    """
    # Output directory and filename
    output_dir = "data"
    output_filename = "result.csv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, output_filename)

    # Cosmological parameters (units in comments)
    H0 = 67.5                # Hubble constant [km/s/Mpc]
    ombh2 = 0.022            # Baryon density Omega_b h^2 [dimensionless]
    omch2 = 0.122            # Cold dark matter density Omega_c h^2 [dimensionless]
    mnu = 0.06               # Neutrino mass sum [eV]
    omk = 0.0                # Curvature Omega_k [dimensionless]
    tau = 0.06               # Optical depth to reionization [dimensionless]
    r_tensor = 0.1           # Tensor-to-scalar ratio [dimensionless]
    As_scalar = 2e-9         # Scalar amplitude [dimensionless]
    ns_scalar = 0.965        # Scalar spectral index [dimensionless]

    # Delensing efficiency (fraction of lensing B-mode power removed)
    delensing_efficiency = 0.1
    Alens_param = 1.0 - delensing_efficiency  # Scaling of lensing potential power

    # Multipole range
    lmin = 2
    lmax = 3000

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As_scalar, ns=ns_scalar, r=r_tensor)
    pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=1)
    pars.WantTensors = True

    # Run CAMB and get results
    results = camb.get_results(pars)

    # Get partially delensed scalar B-mode spectrum (index 2: BB)
    cl_bb_scalar_delensed = results.get_partially_lensed_cls(
        Alens=Alens_param, lmax=lmax, CMB_unit='muK', raw_cl=True
    )[:, 2]

    # Get tensor B-mode spectrum (index 2: BB)
    cl_bb_tensor = results.get_tensor_cls(
        lmax=lmax, CMB_unit='muK', raw_cl=True
    )[:, 2]

    # Total delensed B-mode spectrum (scalar + tensor)
    cl_bb_total_delensed = cl_bb_scalar_delensed + cl_bb_tensor

    # Prepare output for l=2 to l=3000
    ls = np.arange(lmin, lmax + 1)
    bb_output = cl_bb_total_delensed[lmin:lmax + 1]

    # Save to CSV
    df = pd.DataFrame({'l': ls, 'BB': bb_output})
    df.to_csv(output_path, index=False, float_format='%.8e')

    # Print summary to console
    print("Delensed CMB B-mode power spectrum (C_ell^{BB}) saved to " + output_path)
    print("Columns: l (multipole), BB (delensed C_ell^{BB} in micro-Kelvin^2)")
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())
    print("Total number of rows: " + str(len(df)))

if __name__ == "__main__":
    compute_delensed_BB_spectrum()