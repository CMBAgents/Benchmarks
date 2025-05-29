# filename: codebase/cmb_bmode_power_spectrum.py
import camb
import numpy as np
import os

def compute_cmb_bmode_power_spectrum():
    r"""
    Compute the CMB B-mode polarization power spectrum l(l+1)C_l^{BB}/(2pi) in units of micro-Kelvin squared (uK^2)
    for a flat Lambda CDM cosmology using CAMB, with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (Omega_b h^2): 0.022
        - Cold dark matter density (Omega_c h^2): 0.122
        - Neutrino mass sum (Sigma m_nu): 0.06 eV
        - Curvature (Omega_k): 0
        - Optical depth to reionization (tau): 0.06
        - Tensor-to-scalar ratio (r): 0
        - Scalar amplitude (A_s): 2e-9
        - Scalar spectral index (n_s): 0.965

    The spectrum is computed for multipole moments l=2 to l=3000 and saved to 'data/result.csv' with columns:
        - l: Multipole moment (integer, 2 to 3000)
        - BB: B-mode polarization power spectrum (l(l+1)C_l^{BB}/(2pi) in uK^2)

    Returns
    -------
    None
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cosmological parameters (units in comments)
    H0 = 67.5                # Hubble constant [km/s/Mpc]
    ombh2 = 0.022            # Omega_b h^2 [dimensionless]
    omch2 = 0.122            # Omega_c h^2 [dimensionless]
    mnu = 0.06               # Sum of neutrino masses [eV]
    omk = 0.0                # Curvature [dimensionless]
    tau = 0.06               # Optical depth to reionization [dimensionless]
    As = 2.0e-9              # Scalar amplitude [dimensionless]
    ns = 0.965               # Scalar spectral index [dimensionless]
    r_tensor = 0.0           # Tensor-to-scalar ratio [dimensionless]
    lmax = 3000              # Maximum multipole moment [dimensionless]
    lens_potential_accuracy = 1  # Accuracy parameter for lensing [dimensionless]

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
        r=r_tensor,
        lmax=lmax,
        lens_potential_accuracy=lens_potential_accuracy,
        WantTensors=False
    )

    # Run CAMB and get results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in muK^2 units
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    lensed_scalar_cls = powers['lensed_scalar']  # shape: (lmax+1, 4), columns: TT, EE, BB, TE

    # Extract l and BB spectrum for l=2..3000
    l = np.arange(2, lmax + 1)  # l values [dimensionless]
    BB = lensed_scalar_cls[2:lmax + 1, 2]  # BB spectrum [uK^2]

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    header_str = "l,BB"
    np.savetxt(output_path, np.column_stack((l, BB)), delimiter=",", header=header_str, comments="")

    # Print summary to console
    print("CMB B-mode power spectrum (lensed scalar, r=0) saved to " + output_path)
    print("Columns: l (multipole moment, 2-3000), BB (l(l+1)C_l^{BB}/(2pi) in uK^2)")
    print("First 5 rows:")
    for i in range(5):
        print("l = " + str(l[i]) + ", BB = " + str(BB[i]) + " uK^2")
    print("Total rows saved: " + str(len(l)))


if __name__ == "__main__":
    compute_cmb_bmode_power_spectrum()