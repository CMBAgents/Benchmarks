# filename: codebase/compute_cmb_bb.py
import camb
import numpy as np
import os

def compute_cmb_bb_spectrum(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density [dimensionless]
    omch2=0.122,            # Cold dark matter density [dimensionless]
    mnu=0.06,               # Neutrino mass sum [eV]
    omk=0.0,                # Curvature [dimensionless]
    tau=0.06,               # Optical depth to reionization [dimensionless]
    As=2e-9,                # Scalar amplitude [dimensionless]
    ns=0.965,               # Scalar spectral index [dimensionless]
    r=0.1,                  # Tensor-to-scalar ratio [dimensionless]
    lmax=3000,              # Maximum multipole moment [dimensionless]
    output_csv='data/result.csv' # Output CSV file path
):
    """
    Compute the CMB B-mode polarization power spectrum (C_l^{BB}) for the specified cosmology
    and save the results to a CSV file.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Baryon density parameter (Omega_b h^2).
    omch2 : float
        Cold dark matter density parameter (Omega_c h^2).
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter (Omega_k).
    tau : float
        Optical depth to reionization.
    As : float
        Scalar amplitude.
    ns : float
        Scalar spectral index.
    r : float
        Tensor-to-scalar ratio.
    lmax : int
        Maximum multipole moment to compute.
    output_csv : str
        Path to the output CSV file.

    Returns
    -------
    None
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
    # Set lmax and lensing accuracy for accurate lensed B-modes
    pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=1)

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Get total (lensed scalar + tensor) CMB power spectra in muK^2, raw Cl
    # Output: array of shape (lmax+1, 4), columns: TT, EE, BB, TE
    cls = results.get_total_cls(lmax=lmax, CMB_unit='muK', raw_cl=True)

    # Extract multipole moments and BB spectrum
    l = np.arange(2, lmax + 1)  # l: 2 to lmax
    BB = cls[2:lmax + 1, 2]     # BB: C_l^{BB} in muK^2

    # Save to CSV
    data = np.column_stack((l, BB))
    header = "l,BB"
    np.savetxt(output_csv, data, delimiter=",", header=header, comments='')

    # Print summary and first few rows for verification
    print("CMB B-mode power spectrum (C_l^{BB}) computed for l=2 to l=" + str(lmax) + ".")
    print("Results saved to " + output_csv)
    print("First 5 rows (l, BB [muK^2]):")
    for i in range(min(5, len(l))):
        print(str(int(l[i])) + ", " + str(BB[i]))

if __name__ == "__main__":
    compute_cmb_bb_spectrum()