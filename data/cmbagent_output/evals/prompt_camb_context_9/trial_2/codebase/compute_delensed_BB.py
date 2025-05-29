# filename: codebase/compute_delensed_BB.py
import camb
import numpy as np
import pandas as pd
import os

def compute_delensed_BB_spectrum(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density [dimensionless]
    omch2=0.122,            # Cold dark matter density [dimensionless]
    mnu=0.06,               # Sum of neutrino masses [eV]
    omk=0.0,                # Curvature [dimensionless]
    tau=0.06,               # Optical depth to reionization [dimensionless]
    As=2e-9,                # Scalar amplitude [dimensionless]
    ns=0.965,               # Scalar spectral index [dimensionless]
    r=0.1,                  # Tensor-to-scalar ratio [dimensionless]
    delensing_efficiency=0.1, # Delensing efficiency [fraction]
    lmin=2,                 # Minimum multipole
    lmax=3000,              # Maximum multipole
    output_csv='data/result.csv' # Output CSV file path
):
    r"""
    Compute the delensed CMB B-mode polarization power spectrum (C_ell^{BB}) for a flat Lambda CDM cosmology
    using CAMB, with specified cosmological and primordial parameters. Applies a given delensing efficiency
    to the scalar B-modes, adds the tensor B-modes, and outputs the total delensed C_ell^{BB} in muK^2 for
    multipoles l=lmin to l=lmax. Results are saved in a CSV file with columns 'l' and 'BB'.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter (Omega_b h^2).
    omch2 : float
        Physical cold dark matter density parameter (Omega_c h^2).
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter (Omega_k).
    tau : float
        Optical depth to reionization.
    As : float
        Scalar amplitude of primordial fluctuations.
    ns : float
        Scalar spectral index.
    r : float
        Tensor-to-scalar ratio.
    delensing_efficiency : float
        Fraction of lensing B-mode power removed (e.g., 0.1 for 10% delensing).
    lmin : int
        Minimum multipole moment to include in output.
    lmax : int
        Maximum multipole moment to include in output.
    output_csv : str
        Path to output CSV file.

    Returns
    -------
    None
        Saves the results to the specified CSV file and prints a summary to the console.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
        r=r,
        WantTensors=True,
        lmax=lmax + 500,  # Use slightly higher lmax for internal accuracy
        lens_potential_accuracy=1
    )

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Delensing: Alens = 1 - delensing_efficiency
    Alens = 1.0 - delensing_efficiency

    # Get partially delensed scalar B-modes (muK^2, raw Cl)
    cls_scalar = results.get_partially_lensed_cls(
        Alens=Alens,
        lmax=lmax,
        CMB_unit='muK',
        raw_cl=True
    )
    cl_bb_scalar = cls_scalar[:, 2]  # BB mode

    # Get tensor B-modes (muK^2, raw Cl)
    cls_tensor = results.get_tensor_cls(
        lmax=lmax,
        CMB_unit='muK',
        raw_cl=True
    )
    cl_bb_tensor = cls_tensor[:, 2]  # BB mode

    # Total delensed B-mode spectrum
    cl_bb_total = cl_bb_scalar + cl_bb_tensor

    # Prepare output for l=lmin to l=lmax
    l_vals = np.arange(lmin, lmax + 1)
    cl_bb_out = cl_bb_total[lmin:lmax + 1]

    # Save to CSV
    df = pd.DataFrame({'l': l_vals.astype(int), 'BB': cl_bb_out})
    df.to_csv(output_csv, index=False)

    # Print summary to console
    print("Delensed CMB B-mode power spectrum (C_ell^{BB}) computed and saved to " + output_csv)
    print("Cosmological parameters used:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  Sum m_nu = " + str(mnu) + " eV")
    print("  Omega_k = " + str(omk))
    print("  tau = " + str(tau))
    print("  As = " + str(As))
    print("  ns = " + str(ns))
    print("  r = " + str(r))
    print("Delensing efficiency: " + str(delensing_efficiency * 100.0) + " %")
    print("Multipole range: l = " + str(lmin) + " to " + str(lmax))
    print("First 5 rows of output:")
    print(df.head())
    print("Last 5 rows of output:")
    print(df.tail())
    print("Total number of multipoles saved: " + str(len(df)))


if __name__ == "__main__":
    compute_delensed_BB_spectrum()