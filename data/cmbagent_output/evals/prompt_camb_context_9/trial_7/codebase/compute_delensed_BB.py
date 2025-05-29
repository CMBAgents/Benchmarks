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
    using CAMB, with specified cosmological parameters and a given delensing efficiency.

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
        Fraction of lensing B-modes removed (e.g., 0.1 for 10% delensing).
    lmin : int
        Minimum multipole moment to include in output.
    lmax : int
        Maximum multipole moment to include in output.
    output_csv : str
        Path to output CSV file.

    Returns
    -------
    None
        Saves the computed delensed B-mode power spectrum to a CSV file with columns:
        'l' : multipole moment (integer)
        'BB' : delensed B-mode power spectrum (C_ell^{BB}) in micro-Kelvin squared (muK^2)
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
        lmax=lmax + 500,  # Slightly higher for accuracy at lmax edge
        lens_potential_accuracy=1
    )

    # Run CAMB
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

    # l values from lmin to lmax
    l_vals = np.arange(lmin, lmax + 1)
    cl_bb_out = cl_bb_total[lmin:lmax + 1]

    # Save to CSV
    df = pd.DataFrame({'l': l_vals.astype(int), 'BB': cl_bb_out})
    df.to_csv(output_csv, index=False)

    # Print summary to console
    print("Delensed CMB B-mode power spectrum (C_ell^{BB}) computed and saved to " + output_csv)
    print("Columns: l (multipole, unitless), BB (delensed C_ell^{BB} in micro-Kelvin^2)")
    print("First 5 rows:")
    print(df.head())
    print("Total rows: " + str(df.shape[0]))
    print("l range: " + str(lmin) + " to " + str(lmax))
    print("BB min: " + str(np.min(cl_bb_out)) + " muK^2, BB max: " + str(np.max(cl_bb_out)) + " muK^2")


if __name__ == "__main__":
    compute_delensed_BB_spectrum()