# filename: codebase/delensed_cmb.py
import camb
import numpy as np
import os
import csv

def compute_delensed_cmb_tt_spectrum(
    H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0.0, tau=0.06,
    As=2.0e-9, ns=0.965, delensing_efficiency=0.8, lmin=2, lmax=3000,
    lmax_calc=3500, output_dir="data", output_filename="result.csv"
):
    r"""
    Compute the delensed CMB temperature power spectrum D_l^{TT} for given cosmological parameters
    and save the result to a CSV file.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density Omega_b h^2.
    omch2 : float
        Physical cold dark matter density Omega_c h^2.
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter Omega_k.
    tau : float
        Optical depth to reionization.
    As : float
        Scalar amplitude.
    ns : float
        Scalar spectral index.
    delensing_efficiency : float
        Fraction of lensing power removed (0.8 for 80% delensing).
    lmin : int
        Minimum multipole to include in output (inclusive).
    lmax : int
        Maximum multipole to include in output (inclusive).
    lmax_calc : int
        Maximum multipole for internal CAMB calculation (should be >= lmax).
    output_dir : str
        Directory to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax=lmax_calc, lens_potential_accuracy=1)
    pars.WantScalars = True
    pars.DoLensing = True
    pars.WantTensors = False

    # Run CAMB
    results = camb.get_results(pars)

    # Compute delensed CMB power spectra
    Alens_param = 1.0 - delensing_efficiency  # 0.2 for 80% delensing
    delensed_cls = results.get_partially_lensed_cls(
        Alens=Alens_param, lmax=lmax, CMB_unit='muK', raw_cl=False
    )

    # Extract l and D_l^{TT}
    ls = np.arange(lmin, lmax + 1)
    dl_TT_delensed = delensed_cls[lmin:lmax+1, 0]  # D_l = l(l+1)C_l/(2pi), muK^2

    # Save to CSV
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['l', 'TT'])
        for l_val, tt_val in zip(ls, dl_TT_delensed):
            writer.writerow([l_val, tt_val])

    # Print summary to console
    print("Delensed CMB TT power spectrum (D_l^{TT} = l(l+1)C_l^{TT}/(2pi), muK^2) saved to " + output_path)
    print("Multipole range: l = " + str(lmin) + " to " + str(lmax))
    print("First 5 rows:")
    for i in range(min(5, len(ls))):
        print("l = " + str(ls[i]) + ", TT = " + str(dl_TT_delensed[i]) + " muK^2")
    print("Total rows saved: " + str(len(ls)))


if __name__ == "__main__":
    compute_delensed_cmb_tt_spectrum()