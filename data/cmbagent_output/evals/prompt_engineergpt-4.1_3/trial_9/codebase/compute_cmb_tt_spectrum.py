# filename: codebase/compute_cmb_tt_spectrum.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

def compute_cmb_tt_spectrum():
    r"""
    Compute the CMB temperature power spectrum (C_l^{TT}) for a flat Lambda CDM cosmology
    using CAMB with the specified parameters. The results are saved in a CSV file.

    Returns
    -------
    None
        The function saves the results to 'data/result.csv' and prints a summary to the console.

    Units:
    -------
    - H0: km/s/Mpc
    - ombh2: dimensionless (Ω_b h^2)
    - omch2: dimensionless (Ω_c h^2)
    - mnu: eV
    - omk: dimensionless (Ω_k)
    - tau: dimensionless (optical depth)
    - As: dimensionless (scalar amplitude)
    - ns: dimensionless (scalar spectral index)
    - l: integer (multipole moment)
    - TT: μK^2 (temperature power spectrum)
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cosmological parameters
    H0 = 74.0  # km/s/Mpc
    ombh2 = 0.022  # Ω_b h^2
    omch2 = 0.122  # Ω_c h^2
    mnu = 0.06  # eV
    omk = 0.0  # Ω_k
    tau = 0.06  # optical depth
    As = 2e-9  # scalar amplitude
    ns = 0.965  # scalar spectral index

    # Multipole range
    lmin = 2
    lmax = 3000

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.WantTensors = False
    pars.Want_CMB = True

    # Run CAMB
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
    cl = powers['unlensed_scalar']  # shape: (lmax+1, 4), columns: TT, EE, BB, TE

    # Extract TT spectrum for l=2 to l=3000
    ells = np.arange(cl.shape[0])  # l=0 to lmax
    mask = (ells >= lmin) & (ells <= lmax)
    l_vals = ells[mask]
    TT_vals = cl[mask, 0]  # TT column, units: μK^2

    # Save to CSV
    df = pd.DataFrame({'l': l_vals, 'TT': TT_vals})
    csv_path = os.path.join(output_dir, "result.csv")
    df.to_csv(csv_path, index=False)

    # Print summary
    pd.set_option("display.precision", 6)
    pd.set_option("display.width", 120)
    print("CMB TT power spectrum (μK^2) for l=2 to l=3000 saved to data/result.csv")
    print("First 5 rows:\n", df.head())
    print("Last 5 rows:\n", df.tail())
    print("Total number of multipoles: " + str(len(df)))


if __name__ == "__main__":
    compute_cmb_tt_spectrum()