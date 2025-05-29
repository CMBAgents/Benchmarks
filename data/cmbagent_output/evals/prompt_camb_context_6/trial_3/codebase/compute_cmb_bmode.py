# filename: codebase/compute_cmb_bmode.py
import camb
import numpy as np
import os

def compute_cmb_bmode_power_spectrum():
    r"""
    Compute the CMB raw B-mode polarization power spectrum (C_l^{BB}) for a flat Lambda CDM cosmology
    using CAMB with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Neutrino mass sum (mnu): 0.06 eV
        - Curvature (omk): 0
        - Optical depth to reionization (tau): 0.06
        - Tensor-to-scalar ratio (r): 0.1
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965

    The function computes the B-mode power spectrum (C_l^{BB}) in units of micro-Kelvin squared (μK^2)
    for multipole moments from l=2 to l=3000, and saves the results in a CSV file named 'result.csv'
    in the 'data/' directory with two columns:
        - l: Multipole moment (integer values from 2 to 3000)
        - BB: B-mode polarization power spectrum (C_l^{BB} in μK^2)
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cosmological parameters
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
    # Set lmax and lensing accuracy
    pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=1)

    # Run CAMB
    results = camb.get_results(pars)

    # Get total (lensed scalar + tensor) CMB power spectra in μK^2, raw Cl
    cls = results.get_total_cls(lmax=lmax, CMB_unit='muK', raw_cl=True)

    # Extract l and BB spectrum (index 2)
    l = np.arange(2, lmax + 1)  # Multipole moments [dimensionless]
    BB = cls[2:lmax + 1, 2]     # B-mode power spectrum [μK^2]

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    header = "l,BB"
    data = np.column_stack((l, BB))
    np.savetxt(output_path, data, delimiter=",", header=header, comments="")

    # Print summary to console
    print("CMB B-mode power spectrum (C_l^{BB}) computed for l=2 to l=3000.")
    print("Results saved to " + output_path)
    print("First 5 rows:")
    for i in range(min(5, len(l))):
        print("l = " + str(l[i]) + ", BB = " + str(BB[i]) + " μK^2")


if __name__ == "__main__":
    compute_cmb_bmode_power_spectrum()