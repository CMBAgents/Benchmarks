# filename: codebase/compute_delensed_BB_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_delensed_BB_spectrum(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density Omega_b h^2
    omch2=0.122,            # Cold dark matter density Omega_c h^2
    mnu=0.06,               # Sum of neutrino masses [eV]
    omk=0.0,                # Curvature Omega_k
    tau=0.06,               # Optical depth to reionization
    As=2.0e-9,              # Scalar amplitude
    ns=0.965,               # Scalar spectral index
    r_tensor=0.1,           # Tensor-to-scalar ratio
    delensing_efficiency=0.10, # Fraction of lensing B-modes removed
    output_lmin=2,          # Minimum multipole
    output_lmax=3000,       # Maximum multipole
    calculation_lmax=3100,  # lmax for internal calculation (for accuracy)
    output_dir="data/",     # Output directory
    output_filename="result.csv" # Output CSV filename
):
    r"""
    Compute the raw delensed CMB B-mode polarization power spectrum (C_ell^{BB}) for a flat Lambda CDM cosmology
    using CAMB, with specified cosmological parameters and a given delensing efficiency.

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
        Scalar amplitude.
    ns : float
        Scalar spectral index.
    r_tensor : float
        Tensor-to-scalar ratio.
    delensing_efficiency : float
        Fraction of lensing B-mode power removed (e.g., 0.10 for 10% delensing).
    output_lmin : int
        Minimum multipole moment to output.
    output_lmax : int
        Maximum multipole moment to output.
    calculation_lmax : int
        lmax for internal calculation (should be >= output_lmax for accuracy).
    output_dir : str
        Directory to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
        Saves the delensed B-mode power spectrum to a CSV file with columns 'l' and 'BB'.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=r_tensor)
    pars.WantTensors = True
    pars.set_for_lmax(lmax=calculation_lmax, lens_potential_accuracy=1)

    # Compute results
    results = camb.get_results(pars)

    # Delensing: Alens = 1 - delensing_efficiency
    Alens_param = 1.0 - delensing_efficiency

    # Get delensed scalar B-modes (lensed scalar BB with reduced lensing)
    cls_delensed_scalar = results.get_partially_lensed_cls(
        Alens=Alens_param,
        lmax=output_lmax,
        CMB_unit='muK',
        raw_cl=True
    )
    cl_bb_delensed_scalar = cls_delensed_scalar[:, 2]  # BB is column 2

    # Get tensor B-modes
    cls_tensor = results.get_tensor_cls(
        lmax=output_lmax,
        CMB_unit='muK',
        raw_cl=True
    )
    cl_bb_tensor = cls_tensor[:, 2]  # BB is column 2

    # Total delensed BB = delensed scalar BB + tensor BB
    total_cl_bb = cl_bb_delensed_scalar + cl_bb_tensor

    # Prepare output for l=output_lmin to output_lmax
    ls = np.arange(output_lmin, output_lmax + 1)
    bb_spectrum_output = total_cl_bb[output_lmin:output_lmax + 1]

    # Save to CSV
    output_path = os.path.join(output_dir, output_filename)
    output_df = pd.DataFrame({'l': ls, 'BB': bb_spectrum_output})
    output_df.to_csv(output_path, index=False)

    # Print detailed output information
    print("Delensed CMB B-mode power spectrum (C_ell^{BB}) saved to " + output_path)
    print("Columns: l (multipole, dimensionless), BB (delensed C_ell^{BB} in microkelvin^2 [μK^2])")
    print("First 10 rows:")
    print(output_df.head(10))
    print("Last 10 rows:")
    print(output_df.tail(10))
    print("Total number of multipoles saved: " + str(len(output_df)))


if __name__ == "__main__":
    compute_delensed_BB_spectrum()