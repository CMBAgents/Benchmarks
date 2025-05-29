# filename: codebase/compute_cmb_spectrum.py
import os
import numpy as np
import pandas as pd

# Ensure output directory exists
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    import camb
    from camb import model, initialpower
except ImportError as e:
    print("CAMB package is required but not installed. Please install 'camb' and rerun the code.")
    raise e


def compute_cmb_tt_spectrum(
    H0=67.3,           # Hubble constant [km/s/Mpc]
    ombh2=0.022,       # Omega_b h^2 [dimensionless]
    omch2=0.122,       # Omega_c h^2 [dimensionless]
    mnu=0.06,          # Sum of neutrino masses [eV]
    omk=0.05,          # Curvature Omega_k [dimensionless]
    tau=0.06,          # Optical depth to reionization [dimensionless]
    As=2e-9,           # Scalar amplitude [dimensionless]
    ns=0.965,          # Scalar spectral index [dimensionless]
    lmin=2,            # Minimum multipole
    lmax=3000          # Maximum multipole
):
    r"""
    Compute the CMB temperature power spectrum for given cosmological parameters.

    Returns:
        l (ndarray): Multipole moments (l)
        TT (ndarray): Temperature power spectrum l(l+1)C_l^{TT}/(2pi) in microK^2
    """
    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.WantTensors = False
    pars.Want_CMB = True

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total'])
    totCL = powers['total']

    # l values: CAMB returns from l=0,1,...,lmax
    ell = np.arange(totCL.shape[0])
    # Only keep l in [lmin, lmax]
    mask = (ell >= lmin) & (ell <= lmax)
    ell = ell[mask]
    # TT spectrum: column 0 is TT
    cltt = totCL[mask, 0]  # units: muK^2

    # Compute l(l+1)C_l/(2pi)
    tt_spectrum = ell * (ell + 1) * cltt / (2.0 * np.pi)  # units: muK^2

    return ell, tt_spectrum


if __name__ == "__main__":
    # Compute the spectrum
    l, TT = compute_cmb_tt_spectrum()

    # Save to CSV
    df = pd.DataFrame({'l': l, 'TT': TT})
    csv_path = os.path.join("data", "result.csv")
    df.to_csv(csv_path, index=False)

    # Print summary
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print("CMB temperature power spectrum (l(l+1)C_l^{TT}/(2pi)) computed for non-flat Lambda CDM cosmology.")
    print("Parameters used:")
    print("  H0 = 67.3 km/s/Mpc")
    print("  Omega_b h^2 = 0.022")
    print("  Omega_c h^2 = 0.122")
    print("  Sum m_nu = 0.06 eV")
    print("  Omega_k = 0.05")
    print("  tau = 0.06")
    print("  A_s = 2e-9")
    print("  n_s = 0.965")
    print("Multipole range: l = 2 to 3000")
    print("Results saved to: " + csv_path)
    print("First 10 rows of the result:")
    print(df.head(10).to_string(index=False))
    print("Last 10 rows of the result:")
    print(df.tail(10).to_string(index=False))