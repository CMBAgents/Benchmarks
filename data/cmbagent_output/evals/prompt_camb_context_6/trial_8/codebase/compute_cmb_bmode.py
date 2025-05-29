# filename: codebase/compute_cmb_bmode.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_bmode_power_spectrum():
    r"""
    Compute the raw CMB B-mode polarization power spectrum (C_l^{BB}) for a flat Lambda CDM cosmology
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
    The B-mode power spectrum (C_l^{BB}) is computed in units of microkelvin squared (muK^2)
    for multipole moments l=2 to l=3000. The results are saved in 'data/result.csv' with columns:
        - l: Multipole moment (integer, 2 to 3000)
        - BB: B-mode polarization power spectrum (C_l^{BB} in muK^2)
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
    lmax = 3000              # Maximum multipole moment [integer]

    # Set up CAMB parameters
    pars = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu,
        omk=omk,
        tau=tau,
        As=As,
        ns=ns,
        r=r_tensor,
        lmax=lmax,
        lens_potential_accuracy=1,
        halofit_version='mead'
    )
    pars.WantTensors = True

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Get total (lensed scalar + tensor) CMB power spectra in muK^2, raw Cl
    total_cls = results.get_total_cls(lmax=lmax, CMB_unit='muK', raw_cl=True)
    # total_cls shape: (lmax+1, 4), columns: TT, EE, BB, TE

    # Extract l and BB spectrum for l=2..lmax
    ls = np.arange(2, lmax + 1)  # l values
    cl_BB = total_cls[2:lmax + 1, 2]  # BB spectrum in muK^2

    # Prepare output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'l': ls, 'BB': cl_BB})
    df.to_csv(output_path, index=False, header=['l', 'BB'])

    # Print summary of results
    print("CMB B-mode polarization power spectrum (C_l^{BB}) computed and saved to data/result.csv")
    print("Cosmological parameters used:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  ombh2 = " + str(ombh2))
    print("  omch2 = " + str(omch2))
    print("  mnu = " + str(mnu) + " eV")
    print("  omk = " + str(omk))
    print("  tau = " + str(tau))
    print("  r = " + str(r_tensor))
    print("  As = " + str(As))
    print("  ns = " + str(ns))
    print("  lmax = " + str(lmax))
    print("Output columns:")
    print("  l  (multipole moment, 2 to 3000)")
    print("  BB (C_l^{BB} in microkelvin^2)")
    print("First 5 rows of output:")
    print(df.head().to_string(index=False))
    print("Last 5 rows of output:")
    print(df.tail().to_string(index=False))


if __name__ == "__main__":
    compute_cmb_bmode_power_spectrum()