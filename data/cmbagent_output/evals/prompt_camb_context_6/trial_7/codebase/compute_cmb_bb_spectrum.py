# filename: codebase/compute_cmb_bb_spectrum.py
import camb
import numpy as np
import os

def compute_cmb_bb_spectrum(
    H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0.0, tau=0.06,
    As=2e-9, ns=0.965, r=0.1, lmax=3000, output_csv='data/result.csv'
):
    r"""
    Compute the CMB B-mode polarization power spectrum (C_l^{BB}) for a flat Lambda CDM cosmology
    using CAMB, and save the results for l=2 to lmax in a CSV file.

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
    output_csv : str
        Path to output CSV file.

    Returns
    -------
    None
        Saves the results to output_csv.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
    # Get total (lensed scalar + tensor) CMB power spectra in muK^2, as raw C_l
    cls = results.get_total_cls(lmax=lmax, CMB_unit='muK', raw_cl=True)
    # Extract l and BB spectrum (l=2 to lmax)
    l_arr = np.arange(2, lmax + 1)
    BB_arr = cls[2:lmax + 1, 2]
    # Save to CSV
    data = np.column_stack([l_arr, BB_arr])
    header = "l,BB"
    np.savetxt(output_csv, data, delimiter=",", header=header, comments="")
    # Print summary
    print("CMB B-mode power spectrum (C_l^{BB}) computed for l=2 to l=" + str(lmax) + ".")
    print("Results saved to " + output_csv)
    print("First 5 rows (l, BB [muK^2]):")
    for i in range(min(5, data.shape[0])):
        print(int(data[i,0]), data[i,1])


if __name__ == "__main__":
    compute_cmb_bb_spectrum()