# filename: codebase/compute_delensed_BB_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_delensed_BB_spectrum(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density Omega_b h^2
    omch2=0.122,            # Cold dark matter density Omega_c h^2
    mnu=0.06,               # Neutrino mass sum [eV]
    omk=0.0,                # Curvature Omega_k
    tau=0.06,               # Optical depth to reionization
    As=2e-9,                # Scalar amplitude
    ns=0.965,               # Scalar spectral index
    r=0.1,                  # Tensor-to-scalar ratio
    lmax=3000,              # Maximum multipole
    delensing_efficiency=0.1, # Delensing efficiency (fraction of lensing removed)
    lens_potential_accuracy=2, # Lensing accuracy
    output_csv_path="data/result.csv" # Output CSV file path
):
    r"""
    Compute the raw delensed CMB B-mode polarization power spectrum (C_ell^{BB}) for a flat Lambda CDM cosmology
    using CAMB, with specified cosmological and primordial parameters, and a given delensing efficiency.
    The result is saved as a CSV file with columns 'l' and 'BB' (in micro-Kelvin squared) for l=2 to lmax.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density Omega_b h^2.
    omch2 : float
        Physical cold dark matter density Omega_c h^2.
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter Omega_k.
    tau : float
        Optical depth to reionization.
    As : float
        Scalar amplitude of primordial power spectrum.
    ns : float
        Scalar spectral index.
    r : float
        Tensor-to-scalar ratio.
    lmax : int
        Maximum multipole moment to compute.
    delensing_efficiency : float
        Fraction of lensing B-mode power removed (e.g., 0.1 for 10% delensing).
    lens_potential_accuracy : int
        Accuracy setting for lensing potential calculation.
    output_csv_path : str
        Path to save the output CSV file.

    Returns
    -------
    None
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu,
        omk=omk,
        tau=tau
    )
    pars.InitPower.set_params(
        As=As,
        ns=ns,
        r=r
    )
    pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=lens_potential_accuracy)
    pars.WantTensors = True

    # Run CAMB
    results = camb.get_results(pars)

    # Compute delensed scalar B-modes (Alens = 1 - delensing_efficiency)
    Alens_factor = 1.0 - delensing_efficiency
    cls_partially_lensed_scalar = results.get_partially_lensed_cls(
        Alens=Alens_factor,
        CMB_unit='muK',
        raw_cl=True,
        lmax=lmax
    )
    # BB is index 2
    C_BB_scalar_delensed = cls_partially_lensed_scalar[:, 2]

    # Get tensor B-modes (primordial, not affected by delensing)
    tensor_cls = results.get_tensor_cls(
        CMB_unit='muK',
        raw_cl=True,
        lmax=lmax
    )
    C_BB_tensor = tensor_cls[:, 2]

    # Total delensed B-mode power spectrum
    C_BB_total_delensed = C_BB_scalar_delensed + C_BB_tensor

    # Prepare output for l=2 to lmax
    ls = np.arange(lmax + 1)
    ls_output = ls[2:lmax + 1]
    BB_output = C_BB_total_delensed[2:lmax + 1]

    # Save to CSV
    df = pd.DataFrame({'l': ls_output, 'BB': BB_output})
    df.to_csv(output_csv_path, index=False)

    # Print summary
    print("Delensed CMB B-mode power spectrum (C_ell^{BB}) computed and saved to " + output_csv_path)
    print("Columns: l (multipole, 2 to " + str(lmax) + "), BB (delensed C_ell^{BB} in micro-Kelvin^2)")
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())

if __name__ == "__main__":
    compute_delensed_BB_spectrum()
