# filename: codebase/compute_cmb_ee_spectrum.py
import numpy as np
import pandas as pd
import camb
from camb import model, initialpower

def compute_cmb_ee_spectrum():
    r"""
    Computes the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) for a flat Lambda CDM cosmology
    using CAMB, with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Scalar amplitude (As): 1.8e-9 * exp(2 * tau)
        - Scalar spectral index (ns): 0.95
        - Optical depth to reionization (tau): 0.1
        - Reionization model: Exponential reionization with exponent power 2

    Computes the E-mode power spectrum (l(l+1)C_l^{EE}/(2pi)) in units of microkelvin^2 for multipole moments
    from l=2 to l=100. Saves the results in a CSV file named 'data/result.csv' with columns:
        - l: Multipole moment (integer values from 2 to 100)
        - EE: E-mode polarization power spectrum (microkelvin^2)
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    tau = 0.1  # Optical depth to reionization [dimensionless]
    ns = 0.95  # Scalar spectral index [dimensionless]
    As = 1.8e-9 * np.exp(2 * tau)  # Scalar amplitude [dimensionless]

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(100, lens_potential_accuracy=0)

    # Set exponential reionization with exponent power 2
    pars.Reion.use_optical_depth = True
    pars.Reion.reionization = True
    pars.Reion.reionization_scheme = camb.reionization.ReionizationScheme.Exp
    pars.Reion.exp_reion_exponent = 2.0

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=100)
    EE = powers['total'][:,2]  # EE spectrum in muK^2

    # l values
    l = np.arange(EE.shape[0])
    # Only keep l=2 to l=100
    lmin = 2
    lmax = 100
    l_range = np.arange(lmin, lmax+1)
    EE_range = EE[lmin:lmax+1]

    # Compute l(l+1)C_l^{EE}/(2pi)
    factor = l_range * (l_range + 1) / (2.0 * np.pi)
    EE_power = factor * EE_range  # [muK^2]

    # Save to CSV
    df = pd.DataFrame({'l': l_range, 'EE': EE_power})
    output_path = 'data/result.csv'
    df.to_csv(output_path, index=False)

    # Print results to console
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', lambda x: '%.6e' % x)
    print("CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) in microkelvin^2 for l=2 to l=100:")
    print(df)

if __name__ == "__main__":
    compute_cmb_ee_spectrum()
