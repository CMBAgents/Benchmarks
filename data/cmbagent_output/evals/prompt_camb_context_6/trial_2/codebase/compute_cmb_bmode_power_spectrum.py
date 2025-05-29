# filename: codebase/compute_cmb_bmode_power_spectrum.py
import camb
import numpy as np
import os

def compute_cmb_bmode_power_spectrum():
    r"""
    Compute the CMB raw B-mode polarization power spectrum (C_l^{BB}) for a flat Lambda CDM cosmology
    using CAMB, with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Neutrino mass sum (mnu): 0.06 eV
        - Curvature (omk): 0
        - Optical depth to reionization (tau): 0.06
        - Tensor-to-scalar ratio (r): 0.1
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965

    The B-mode power spectrum (C_l^{BB}) is computed in units of microkelvin squared (muK^2)
    for multipole moments l=2 to l=3000. The results are saved in 'data/result.csv' with columns:
        l: Multipole moment (integer, 2 to 3000)
        BB: B-mode polarization power spectrum (C_l^{BB} in muK^2)
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cosmological and primordial parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    As = 2e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]
    r = 0.1  # Tensor-to-scalar ratio [dimensionless]
    lmax = 3000  # Maximum multipole moment

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
    # Set lmax and lensing accuracy for accurate lensed B-modes
    pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=1)

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Get total (lensed scalar + tensor) CMB power spectra in muK^2, raw Cl
    # Output: array of shape (lmax+1, 4), columns: TT, EE, BB, TE
    cls = results.get_total_cls(lmax=lmax, CMB_unit='muK', raw_cl=True)

    # Extract l and BB spectrum for l=2..lmax
    l = np.arange(2, lmax + 1)  # Multipole moments [dimensionless]
    BB = cls[2:lmax + 1, 2]     # B-mode power spectrum [muK^2]

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    header = "l,BB"
    data = np.column_stack((l, BB))
    np.savetxt(output_path, data, delimiter=",", header=header, comments="")

    # Print summary to console
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print("CMB B-mode power spectrum (C_l^{BB}) computed for l=2 to l=3000.")
    print("Results saved to " + output_path)
    print("First 10 rows (l, BB [muK^2]):")
    print(data[:10])

if __name__ == "__main__":
    compute_cmb_bmode_power_spectrum()
