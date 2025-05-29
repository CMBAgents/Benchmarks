# filename: codebase/compute_delensed_BB.py
import camb
import numpy as np
import pandas as pd
import os

def compute_delensed_BB_spectrum(
    H0=67.5,
    ombh2=0.022,
    omch2=0.122,
    mnu=0.06,
    omk=0.0,
    tau=0.06,
    r=0.1,
    As=2e-9,
    ns=0.965,
    delensing_efficiency=0.1,
    lmax=3000,
    output_folder="data/",
    output_filename="result.csv"
):
    r"""
    Compute the raw delensed CMB B-mode polarization power spectrum (C_l^{BB}) for a flat Lambda CDM cosmology.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density, Omega_b h^2.
    omch2 : float
        Physical cold dark matter density, Omega_c h^2.
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter Omega_k.
    tau : float
        Optical depth to reionization.
    r : float
        Tensor-to-scalar ratio.
    As : float
        Scalar amplitude.
    ns : float
        Scalar spectral index.
    delensing_efficiency : float
        Fraction of lensing B-mode power removed (e.g., 0.1 for 10% delensing).
    lmax : int
        Maximum multipole moment to compute.
    output_folder : str
        Folder to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
        Saves the results to a CSV file with columns 'l' and 'BB' (in muK^2).
    """
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=r)
    pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=1)
    pars.WantTensors = True

    # Run CAMB
    results = camb.get_results(pars)

    # Get partially delensed scalar BB spectrum (Alens = 1 - delensing_efficiency)
    Alens = 1.0 - delensing_efficiency
    cl_lensed = results.get_partially_lensed_cls(
        Alens=Alens, lmax=lmax, CMB_unit="muK", raw_cl=True
    )
    cl_bb_scalar_delensed = cl_lensed[:, 2]  # BB is index 2

    # Get tensor BB spectrum
    cl_tensor = results.get_tensor_cls(
        lmax=lmax, CMB_unit="muK", raw_cl=True
    )
    cl_bb_tensor = cl_tensor[:, 2]  # BB is index 2

    # Total delensed BB = (delensed scalar BB) + (tensor BB)
    cl_bb_total_delensed = cl_bb_scalar_delensed + cl_bb_tensor

    # Prepare output for l=2 to lmax
    ls = np.arange(2, lmax + 1)
    bb_output = cl_bb_total_delensed[2:lmax + 1]

    # Save to CSV
    output_path = os.path.join(output_folder, output_filename)
    df = pd.DataFrame({"l": ls, "BB": bb_output})
    df.to_csv(output_path, index=False, float_format="%.8e")

    # Print summary
    print("Delensed CMB B-mode power spectrum (C_l^{BB}) saved to " + output_path)
    print("Columns: l (multipole), BB (muK^2)")
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())
    print("Total number of multipoles: " + str(len(df)))


if __name__ == "__main__":
    compute_delensed_BB_spectrum()