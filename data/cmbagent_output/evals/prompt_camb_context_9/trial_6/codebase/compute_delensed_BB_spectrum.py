# filename: codebase/compute_delensed_BB_spectrum.py
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
    As=2.0e-9,
    ns=0.965,
    r=0.1,
    delensing_efficiency=0.1,
    lmin=2,
    lmax=3000,
    output_folder="data/",
    output_filename="result.csv"
):
    """
    Compute the delensed CMB B-mode polarization power spectrum (C_ell^{BB}) for a flat Lambda CDM cosmology
    using CAMB, with specified cosmological parameters and a given delensing efficiency.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter, Omega_b h^2.
    omch2 : float
        Physical cold dark matter density parameter, Omega_c h^2.
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter (0 for flat universe).
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
        Minimum multipole moment to include in output (inclusive).
    lmax : int
        Maximum multipole moment to include in output (inclusive).
    output_folder : str
        Directory to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
        Saves the results to a CSV file and prints a summary to the console.
    """
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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
        lmax=lmax + 500,  # Slightly higher for accuracy at the edge
        lens_potential_accuracy=1
    )

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Delensing: Alens = 1 - delensing_efficiency
    Alens = 1.0 - delensing_efficiency

    # Get partially delensed scalar CMB power spectra (TT, EE, BB, TE)
    cls_scalar = results.get_partially_lensed_cls(
        Alens=Alens,
        lmax=lmax,
        CMB_unit='muK',
        raw_cl=True
    )
    # BB mode is index 2
    cl_bb_scalar = cls_scalar[:, 2]

    # Get tensor CMB power spectra (TT, EE, BB, TE)
    cls_tensor = results.get_tensor_cls(
        lmax=lmax,
        CMB_unit='muK',
        raw_cl=True
    )
    cl_bb_tensor = cls_tensor[:, 2]

    # Total delensed B-mode power spectrum
    cl_bb_total = cl_bb_scalar + cl_bb_tensor

    # Multipole array
    l_array = np.arange(0, lmax + 1)
    l_out = l_array[lmin:]
    cl_bb_out = cl_bb_total[lmin:]

    # Save to CSV
    output_path = os.path.join(output_folder, output_filename)
    df = pd.DataFrame({'l': l_out.astype(int), 'BB': cl_bb_out})
    df.to_csv(output_path, index=False)

    # Print summary
    print("Delensed CMB B-mode power spectrum (C_ell^{BB}) computed and saved to " + output_path)
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
    print("Output multipole range: l = " + str(lmin) + " to " + str(lmax))
    print("First 5 rows of output:")
    print(df.head())
    print("Total number of rows: " + str(df.shape[0]))


if __name__ == "__main__":
    compute_delensed_BB_spectrum()