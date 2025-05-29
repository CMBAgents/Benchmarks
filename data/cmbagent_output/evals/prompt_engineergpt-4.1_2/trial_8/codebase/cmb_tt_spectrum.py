# filename: codebase/cmb_tt_spectrum.py
import os
import numpy as np
import pandas as pd

# Ensure data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

try:
    import camb
    from camb import model, initialpower
except ImportError as e:
    print("CAMB is not installed. Please install camb and rerun the code.")
    raise e


def compute_cmb_tt_spectrum():
    r"""
    Compute the CMB temperature power spectrum (C_l^{TT}) for a flat Lambda CDM cosmology
    using CAMB, with the following parameters:
        - Hubble constant (H0): 70 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Neutrino mass sum (mnu): 0.06 eV
        - Curvature (omk): 0
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965

    Computes the temperature power spectrum (C_l^{TT}) in units of microkelvin^2
    for multipole moments l=2 to l=3000, and saves the result to 'data/result.csv'.

    Returns
    -------
    None
    """
    # Cosmological parameters
    H0 = 70.0  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    As = 2e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]

    # Multipole range
    lmin = 2
    lmax = 3000

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.WantTensors = False

    # Calculate results for these parameters
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
    # 'total' contains the raw Cls (no lensing, no foregrounds)
    cl_tot = powers['total']

    # cl_tot shape: (lmax+1, 4) columns: TT, EE, BB, TE
    # l values: 0 to lmax
    # Extract l and TT
    ells = np.arange(cl_tot.shape[0])  # l=0 to lmax
    # Only keep l=2 to lmax
    mask = (ells >= lmin) & (ells <= lmax)
    l_vals = ells[mask]
    # TT spectrum in muK^2
    TT = cl_tot[mask, 0]

    # Save to CSV
    df = pd.DataFrame({'l': l_vals, 'TT': TT})
    output_path = os.path.join("data", "result.csv")
    df.to_csv(output_path, index=False)

    # Print summary
    pd.set_option("display.precision", 8)
    pd.set_option("display.max_rows", 10)
    print("CMB temperature power spectrum (C_l^{TT}) in microkelvin^2 saved to data/result.csv")
    print("First and last 5 rows of the result:")
    print(pd.concat([df.head(5), df.tail(5)]))


if __name__ == "__main__":
    compute_cmb_tt_spectrum()
