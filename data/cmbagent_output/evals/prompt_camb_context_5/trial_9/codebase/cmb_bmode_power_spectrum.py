# filename: codebase/cmb_bmode_power_spectrum.py
import camb
import numpy as np
import os

def compute_cmb_bmode_power_spectrum():
    r"""
    Compute the CMB B-mode polarization power spectrum (l(l+1)C_l^{BB}/(2pi)) for a flat Lambda CDM cosmology
    using CAMB, and save the results for l=2..3000 in units of microkelvin^2 to data/result.csv.

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
    r_tensor = 0.0       # Tensor-to-scalar ratio [dimensionless]
    As = 2e-9            # Scalar amplitude [dimensionless]
    ns = 0.965           # Scalar spectral index [dimensionless]
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
        r=r_tensor,
        WantTensors=False
    )
    # Set accuracy for lensed B-modes
    pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=1)

    # Run CAMB
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra (D_l = l(l+1)C_l/(2pi) in microkelvin^2)
    # Output shape: (lmax+1, 4), columns: TT, EE, BB, TE
    powers = results.get_lensed_scalar_cls(lmax=lmax, CMB_unit='muK')

    # Extract l and BB spectrum for l=2..3000
    ls = np.arange(powers.shape[0])  # Multipole moments (unitless)
    BB = powers[:, 2]                # B-mode power spectrum [microkelvin^2]
    ls_out = ls[2:lmax+1]            # l=2..3000
    BB_out = BB[2:lmax+1]            # Corresponding BB values

    # Prepare output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    output_data = np.column_stack((ls_out, BB_out))
    np.savetxt(output_path, output_data, delimiter=",", header="l,BB", comments="")

    # Print concise summary to console
    print("CMB B-mode polarization power spectrum (l(l+1)C_l^{BB}/(2pi)) computed for l=2..3000.")
    print("Results saved to " + output_path)
    print("First 5 rows (l, BB [microkelvin^2]):")
    for i in range(min(5, len(ls_out))):
        print(str(int(ls_out[i])) + ", " + str(BB_out[i]))

if __name__ == "__main__":
    compute_cmb_bmode_power_spectrum()