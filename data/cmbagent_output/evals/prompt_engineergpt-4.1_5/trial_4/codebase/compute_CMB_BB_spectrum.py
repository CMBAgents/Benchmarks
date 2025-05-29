# filename: codebase/compute_CMB_BB_spectrum.py
import os
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

def compute_cmb_bb_spectrum():
    """
    Computes the CMB B-mode polarization power spectrum (l(l+1)C_l^{BB}/(2pi)) for a flat Lambda CDM cosmology
    using CAMB, with the specified parameters. Saves the results in 'data/result.csv'.

    Returns
    -------
    None
    """
    # Ensure data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density parameter [dimensionless]
    omch2 = 0.122  # Cold dark matter density parameter [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.06  # Optical depth to reionization [dimensionless]
    r = 0.0  # Tensor-to-scalar ratio [dimensionless]
    As = 2e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]

    # Multipole range
    lmin = 2
    lmax = 3000

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=r)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.WantTensors = True  # Needed for BB spectrum, even if r=0

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, spectra=['total'])

    # Extract BB spectrum
    cl = powers['total']  # shape: (lmax+1, 4) for TT, EE, BB, TE
    # cl[:,2] is BB in muK^2 units
    ell = np.arange(cl.shape[0])  # l = 0, 1, ..., lmax
    BB = cl[:,2]  # BB spectrum in muK^2

    # Compute l(l+1)C_l^{BB}/(2pi) for l=2 to lmax
    ell_out = np.arange(lmin, lmax+1)
    BB_out = (ell_out * (ell_out + 1) * BB[lmin:lmax+1]) / (2.0 * np.pi)  # [muK^2]

    # Save to CSV
    df = pd.DataFrame({'l': ell_out, 'BB': BB_out})
    csv_path = os.path.join(data_dir, "result.csv")
    df.to_csv(csv_path, index=False)

    # Print summary
    np.set_printoptions(precision=6, suppress=True)
    print("CMB B-mode polarization power spectrum (l(l+1)C_l^{BB}/(2pi)) computed for l=2 to l=3000.")
    print("Units: micro-Kelvin^2")
    print("Results saved to " + csv_path)
    print("First 10 rows:")
    print(df.head(10).to_string(index=False))
    print("Last 10 rows:")
    print(df.tail(10).to_string(index=False))

if __name__ == "__main__":
    compute_cmb_bb_spectrum()