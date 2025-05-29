# filename: codebase/compute_cmb_ee_spectrum.py
import numpy as np
import pandas as pd
import os

# Ensure data directory exists
database_path = "data"
if not os.path.exists(database_path):
    os.makedirs(database_path)

try:
    import camb
    from camb import model, initialpower
except ImportError as e:
    print("CAMB is not installed. Please install camb and rerun the code.")
    raise e


def compute_cmb_ee_spectrum(
    H0=67.5,           # Hubble constant [km/s/Mpc]
    ombh2=0.022,       # Baryon density Omega_b h^2
    omch2=0.122,       # Cold dark matter density Omega_c h^2
    mnu=0.06,          # Neutrino mass sum [eV]
    omk=0.0,           # Curvature Omega_k
    tau=0.04,          # Optical depth to reionization
    As=2e-9,           # Scalar amplitude
    ns=0.965,          # Scalar spectral index
    lmin=2,            # Minimum multipole
    lmax=3000          # Maximum multipole
):
    """
    Compute the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) using CAMB.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Baryon density parameter Omega_b h^2.
    omch2 : float
        Cold dark matter density parameter Omega_c h^2.
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
    lmin : int
        Minimum multipole moment.
    lmax : int
        Maximum multipole moment.

    Returns
    -------
    l : ndarray
        Array of multipole moments (l).
    EE : ndarray
        E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) in microkelvin^2 (uK^2).
    """
    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.WantTensors = False

    # Calculate results for these parameters
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total'])

    # Extract EE spectrum
    totCL = powers['total']
    # totCL columns: TT, EE, BB, TE, [PP, TP] if lensing
    # l runs from 0 to lmax, but first two rows (l=0,1) are zero
    ell = np.arange(totCL.shape[0])  # l = 0, 1, ..., lmax
    EE = totCL[:,1]  # EE column

    # Select l from lmin to lmax
    mask = (ell >= lmin) & (ell <= lmax)
    ell = ell[mask]
    EE = EE[mask]

    # Compute l(l+1)C_l^{EE}/(2pi) in uK^2
    factor = ell * (ell + 1) / (2.0 * np.pi)
    EE_power = factor * EE  # [uK^2]

    return ell, EE_power


if __name__ == "__main__":
    # Compute the spectrum
    l, EE = compute_cmb_ee_spectrum()

    # Save to CSV
    df = pd.DataFrame({'l': l, 'EE': EE})
    output_file = os.path.join(database_path, "result.csv")
    df.to_csv(output_file, index=False)

    # Print summary
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print("CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) computed for l=2 to l=3000.")
    print("Results saved to " + output_file)
    print("First 10 rows:")
    print(df.head(10).to_string(index=False))
    print("\nLast 10 rows:")
    print(df.tail(10).to_string(index=False))
    print("\nColumn units:")
    print("l: Multipole moment (dimensionless)")
    print("EE: E-mode polarization power spectrum (uK^2)")