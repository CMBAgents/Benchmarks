# filename: codebase/compute_cmb_tt_spectrum.py
import camb
import numpy as np
import os

def compute_cmb_tt_spectrum():
    r"""
    Compute the CMB temperature power spectrum (lensed, scalar, TT) for a non-flat Lambda CDM cosmology
    using CAMB, with the following parameters:
        - Hubble constant (H0): 67.3 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Neutrino mass sum (mnu): 0.06 eV
        - Curvature (omk): 0.05
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965
    The spectrum is computed for multipole moments l=2 to l=3000.
    The results are saved to 'data/result.csv' with columns:
        l: Multipole moment (integer, 2 to 3000)
        TT: Temperature power spectrum (l(l+1)C_l^{TT}/(2pi), in microkelvin^2)
    Also prints a summary of the results to the console.
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cosmological parameters
    H0 = 67.3  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.05  # Curvature [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    As = 2.0e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]
    lmax = 3000  # Maximum multipole

    # Set CAMB parameters
    pars = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu,
        omk=omk,
        tau=tau,
        As=As,
        ns=ns,
        lmax=lmax,
        lens_potential_accuracy=1,
        WantScalars=True,
        WantTensors=False,
        DoLensing=True
    )

    # Run CAMB
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra (D_l = l(l+1)C_l/2pi, in microkelvin^2)
    lensed_cls = results.get_lensed_scalar_cls(CMB_unit='muK', lmax=lmax)

    # Extract l and TT spectrum for l=2 to l=3000
    ls = np.arange(2, lmax + 1)  # l values [dimensionless]
    dl_TT = lensed_cls[2:lmax + 1, 0]  # TT spectrum [microkelvin^2]

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    output_data = np.column_stack((ls, dl_TT))
    np.savetxt(output_path, output_data, delimiter=',', header='l,TT', comments='', fmt=['%d', '%.18e'])

    print("CMB temperature power spectrum (lensed, TT) saved to " + output_path)
    print("Columns: l (multipole, dimensionless), TT (l(l+1)C_l^{TT}/(2pi), microkelvin^2)")
    print("\nSample of calculated values:")
    for i in range(min(5, len(ls))):
        print("l = " + str(ls[i]) + ", TT = " + str(dl_TT[i]) + " microkelvin^2")
    if len(ls) > 10:
        print("...")
        for i in range(max(0, len(ls)-5), len(ls)):
            print("l = " + str(ls[i]) + ", TT = " + str(dl_TT[i]) + " microkelvin^2")


if __name__ == "__main__":
    compute_cmb_tt_spectrum()