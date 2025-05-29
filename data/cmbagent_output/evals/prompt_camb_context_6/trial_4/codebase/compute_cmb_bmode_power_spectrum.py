# filename: codebase/compute_cmb_bmode_power_spectrum.py
import camb
import numpy as np
import os

def compute_cmb_bmode_power_spectrum():
    r"""
    Compute the CMB raw B-mode polarization power spectrum (C_l^{BB}) for a flat Lambda CDM cosmology
    using CAMB with specified parameters. The results are saved in 'data/result.csv' with columns:
    l (multipole moment, 2 <= l <= 3000) and BB (B-mode power spectrum in micro-Kelvin squared, μK^2).

    Cosmological parameters:
        H0: Hubble constant [km/s/Mpc]
        ombh2: Baryon density parameter [dimensionless]
        omch2: Cold dark matter density parameter [dimensionless]
        mnu: Sum of neutrino masses [eV]
        omk: Curvature parameter [dimensionless]
        tau: Optical depth to reionization [dimensionless]
        As: Scalar amplitude [dimensionless]
        ns: Scalar spectral index [dimensionless]
        r: Tensor-to-scalar ratio [dimensionless]

    Returns
    -------
    None
    """
    # Output directory and file
    output_dir = "data"
    output_file = os.path.join(output_dir, "result.csv")

    # Ensure output directory exists
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
    # Output: array of shape (lmax+1, 4), columns: TT, EE, BB, TE
    cls = results.get_total_cls(lmax=lmax, CMB_unit='muK', raw_cl=True)

    # Extract BB spectrum for l=2 to l=3000
    ells = np.arange(2, lmax + 1)  # Multipole moments [dimensionless]
    cl_bb = cls[2:lmax + 1, 2]  # B-mode power spectrum [μK^2]

    # Save to CSV
    header = "l,BB"
    data = np.column_stack((ells, cl_bb))
    np.savetxt(output_file, data, delimiter=",", header=header, comments="")

    # Print summary to console
    print("CMB B-mode power spectrum (C_l^{BB}) computed for l=2 to l=3000.")
    print("Results saved to " + output_file)
    print("First 5 rows:")
    for i in range(min(5, data.shape[0])):
        print("l = " + str(int(data[i,0])) + ", BB = " + str(data[i,1]) + " μK^2")

if __name__ == "__main__":
    compute_cmb_bmode_power_spectrum()
