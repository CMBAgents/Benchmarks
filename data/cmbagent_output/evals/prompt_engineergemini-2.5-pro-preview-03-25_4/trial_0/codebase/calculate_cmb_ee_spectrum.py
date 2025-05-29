# filename: codebase/calculate_cmb_ee_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def calculate_cmb_ee_spectrum():
    r"""
    Calculates the CMB E-mode polarization power spectrum D_l^EE = l(l+1)C_l^EE/(2pi)
    for a flat Lambda CDM cosmology using CAMB.

    The cosmological parameters are:
    H0: 67.5 km/s/Mpc (Hubble constant)
    Omega_b h^2: 0.022 (Baryon density parameter)
    Omega_c h^2: 0.122 (Cold dark matter density parameter)
    Sigma m_nu: 0.06 eV (Sum of neutrino masses)
    Omega_k: 0 (Curvature parameter)
    tau: 0.04 (Optical depth to reionization)
    A_s: 2e-9 (Scalar amplitude)
    n_s: 0.965 (Scalar spectral index)

    The spectrum is computed for multipole moments l from 2 to 3000.
    The results (D_l^EE) are in units of muK^2 and saved to 'data/result.csv'.
    """
    # Create data directory if it doesn't exist
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)  # Ensures the directory exists
    
    # Cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Baryon density * h^2
    omch2 = 0.122  # Cold dark matter density * h^2
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0  # Curvature Omega_k
    tau = 0.04  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude at k=0.05 Mpc^-1
    ns = 0.965  # Scalar spectral index
    lmax = 3000  # Maximum multipole moment for output
    
    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    
    # Configure CAMB to calculate spectra up to lmax.
    # lens_potential_accuracy=1 provides a good balance for lensed Cl accuracy.
    # CAMB computes spectra up to the lmax specified here.
    pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=1)
    
    # Calculate results (this runs CAMB)
    results = camb.get_results(pars)
    
    # Get CMB power spectra.
    # 'powers' is a dictionary. powers['total'] contains lensed Cls.
    # The columns are TT, EE, BB, TE.
    # By default (raw_cl=False, CMB_unit='muK^2'), values are D_l = l(l+1)C_l/(2pi) in muK^2.
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK^2', raw_cl=False)
    total_cls = powers['total']  # This is an array: rows are l, columns are TT, EE, BB, TE
                                # total_cls[l,0] is D_l^TT, total_cls[l,1] is D_l^EE, etc.
    
    # Extract l values and EE spectrum for the desired range
    # The array `total_cls` is indexed from l=0 up to lmax.
    # We need results for l from 2 to lmax (inclusive).
    l_values = np.arange(2, lmax + 1, dtype=int)  # Multipole moments from 2 to 3000
    
    # EE spectrum is the second column (index 1) of total_cls.
    # total_cls[l_values, 1] extracts D_l^EE for l in l_values.
    ee_spectrum = total_cls[l_values, 1]  # D_l^EE in muK^2
    
    # Create pandas DataFrame
    df_results = pd.DataFrame({
        'l': l_values,      # Multipole moment l
        'EE': ee_spectrum   # D_l^EE = l(l+1)C_l^EE/(2pi) in muK^2
    })
    
    # Save to CSV file
    file_path = os.path.join(data_dir, 'result.csv')
    df_results.to_csv(file_path, index=False)
    
    print("CMB E-mode polarization power spectrum calculation complete.")
    print("Results saved to: " + file_path)
    print("\nFirst 5 rows of the E-mode power spectrum (l(l+1)C_l^EE/(2pi) in muK^2):")
    print(df_results.head().to_string())
    print("\nLast 5 rows of the E-mode power spectrum (l(l+1)C_l^EE/(2pi) in muK^2):")
    print(df_results.tail().to_string())


if __name__ == '__main__':
    calculate_cmb_ee_spectrum()