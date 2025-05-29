# filename: codebase/cmb_bb_spectrum.py
import camb
import numpy as np
import os

def compute_cmb_bb_spectrum():
    r"""
    Compute the CMB B-mode polarization power spectrum (C_l^{BB}) for a flat Lambda CDM cosmology
    using CAMB, and save the results for l=2 to 3000 in a CSV file.

    Returns
    -------
    None
    """
    # Cosmological parameters (units in comments)
    H0 = 67.5            # Hubble constant [km/s/Mpc]
    ombh2 = 0.022        # Baryon density [dimensionless]
    omch2 = 0.122        # Cold dark matter density [dimensionless]
    mnu = 0.06           # Neutrino mass sum [eV]
    omk = 0.0            # Curvature [dimensionless]
    tau = 0.06           # Optical depth to reionization [dimensionless]
    As = 2e-9            # Scalar amplitude [dimensionless]
    ns = 0.965           # Scalar spectral index [dimensionless]
    r = 0.1              # Tensor-to-scalar ratio [dimensionless]
    lmax = 3000          # Maximum multipole moment

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
        r=r
    )
    # Enable tensor modes for primordial B-modes
    pars.WantTensors = True
    # Set lmax and lensing accuracy
    pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=1)

    # Run CAMB
    results = camb.get_results(pars)

    # Get total (lensed scalar + tensor) CMB power spectra in muK^2, raw Cl
    # Output: CL[0:lmax+1, 0:4] (TT, EE, BB, TE)
    cls = results.get_total_cls(lmax=lmax, CMB_unit='muK', raw_cl=True)

    # Extract l and BB spectrum (l=2 to lmax)
    l_arr = np.arange(2, lmax + 1)
    BB = cls[2:lmax + 1, 2]  # BB is column index 2

    # Prepare output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    header = "l,BB"
    data = np.column_stack((l_arr, BB))
    np.savetxt(output_path, data, delimiter=",", header=header, comments="")

    # Print summary
    print("CMB B-mode power spectrum (C_l^{BB}) calculated for l=2 to 3000.")
    print("Results saved to " + output_path)
    print("First 5 rows:")
    for i in range(min(5, len(l_arr))):
        print("l = " + str(int(l_arr[i])) + ", BB = " + str(BB[i]) + " [muK^2]")


if __name__ == "__main__":
    compute_cmb_bb_spectrum()
