# filename: codebase/compute_cmb_tt_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_tt_spectrum(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.02,             # Baryon density [dimensionless]
    omch2=0.122,            # Cold dark matter density [dimensionless]
    mnu=0.06,               # Sum of neutrino masses [eV]
    omk=0.0,                # Curvature [dimensionless]
    tau=0.06,               # Optical depth to reionization [dimensionless]
    As=2e-9,                # Scalar amplitude [dimensionless]
    ns=0.965,               # Scalar spectral index [dimensionless]
    lmax=3000,              # Maximum multipole moment [dimensionless]
    output_dir="data/",     # Output directory
    output_filename="result.csv" # Output CSV filename
):
    r"""
    Compute the CMB temperature power spectrum (TT) for a flat Lambda CDM cosmology using CAMB.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter \(\Omega_b h^2\) (dimensionless).
    omch2 : float
        Physical cold dark matter density parameter \(\Omega_c h^2\) (dimensionless).
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter \(\Omega_k\) (dimensionless).
    tau : float
        Optical depth to reionization (dimensionless).
    As : float
        Scalar amplitude \(A_s\) (dimensionless).
    ns : float
        Scalar spectral index \(n_s\) (dimensionless).
    lmax : int
        Maximum multipole moment \(l\) to compute (dimensionless).
    output_dir : str
        Directory to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
        Saves the TT power spectrum to a CSV file with columns 'l' and 'TT' (muK^2).
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
        lmax=lmax
    )

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in muK^2 units, up to lmax
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax)
    lensed_scalar_cls = powers['lensed_scalar']  # shape: (lmax+1, 4), columns: TT, EE, BB, TE

    # Extract TT spectrum: D_l^TT = l(l+1)C_l^TT/(2pi) in muK^2
    TT_spectrum = lensed_scalar_cls[:, 0]  # [muK^2], index 0 is TT

    # Multipole moments l = 2 to lmax
    ls = np.arange(2, lmax + 1)
    TT_values = TT_spectrum[ls]  # [muK^2]

    # Save to CSV
    output_path = os.path.join(output_dir, output_filename)
    df = pd.DataFrame({'l': ls, 'TT': TT_values})
    df.to_csv(output_path, index=False)

    # Print summary and sample values
    print("CMB TT power spectrum ($l(l+1)C_l^{TT}/(2\\pi)$ in muK^2) computed for flat Lambda CDM cosmology.")
    print("Cosmological parameters used:")
    print("  H0 = " + str(H0) + " km/s/Mpc")
    print("  Omega_b h^2 = " + str(ombh2))
    print("  Omega_c h^2 = " + str(omch2))
    print("  Sum m_nu = " + str(mnu) + " eV")
    print("  Omega_k = " + str(omk))
    print("  tau = " + str(tau))
    print("  A_s = " + str(As))
    print("  n_s = " + str(ns))
    print("  lmax = " + str(lmax))
    print("Results saved to: " + output_path)
    print("CSV columns: l (multipole moment), TT (muK^2)")
    print("Sample of calculated D_l^TT (muK^2):")
    for l_val in [2, 10, 100, 220, 1000, 2000, 3000]:
        if l_val >= 2 and l_val <= lmax:
            idx = l_val - 2
            print("  l = " + str(l_val) + ", D_l^TT = " + str(TT_values[idx]))
    print("Total number of rows saved: " + str(len(ls)))


if __name__ == "__main__":
    compute_cmb_tt_spectrum()