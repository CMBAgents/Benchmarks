# filename: codebase/camb_cmb_bmode.py
import camb
import numpy as np
import os

def compute_cmb_bmode_spectrum(
    H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0.0, tau=0.06,
    As=2e-9, ns=0.965, r=0.1, lmax=3000, output_folder="data", output_filename="result.csv"
):
    r"""
    Compute the CMB B-mode polarization power spectrum (C_l^{BB}) for a flat Lambda CDM cosmology
    using CAMB and save the results to a CSV file.

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
        Scalar amplitude of primordial power spectrum.
    ns : float
        Scalar spectral index.
    r : float
        Tensor-to-scalar ratio.
    lmax : int
        Maximum multipole moment to compute.
    output_folder : str
        Folder to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
        Saves the results to a CSV file with columns 'l' and 'BB'.
    """
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Set cosmological and primordial parameters
    pars = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu,
        omk=omk,
        tau=tau,
        As=As,
        ns=ns,
        r=r
    )
    # Enable tensor modes for primordial B-modes
    pars.WantTensors = True
    # Set lmax and lensing accuracy
    pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=1)
    # Run CAMB
    results = camb.get_results(pars)
    # Get total (lensed scalar + tensor) CMB power spectra in muK^2, raw Cl
    cls = results.get_total_cls(lmax=lmax, CMB_unit='muK', raw_cl=True)
    # Extract l and BB spectrum (index 2)
    l = np.arange(2, lmax + 1)
    BB = cls[2:lmax + 1, 2]
    # Save to CSV
    output_path = os.path.join(output_folder, output_filename)
    np.savetxt(output_path, np.column_stack([l, BB]), delimiter=',' , header='l,BB', comments='')
    # Print summary
    print("CMB B-mode power spectrum (C_l^{BB}) computed for l=2 to l=" + str(lmax) + ".")
    print("Results saved to " + output_path)
    print("First 5 rows:")
    for i in range(5):
        print("l = " + str(l[i]) + ", BB = " + str(BB[i]) + " [muK^2]")


if __name__ == "__main__":
    compute_cmb_bmode_spectrum()