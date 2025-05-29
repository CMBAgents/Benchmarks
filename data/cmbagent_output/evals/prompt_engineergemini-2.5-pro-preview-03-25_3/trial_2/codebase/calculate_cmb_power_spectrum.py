# filename: codebase/calculate_cmb_power_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def calculate_cmb_power_spectrum():
    r"""
    Calculates the CMB raw temperature power spectrum (C_l^TT) for a specific
    flat Lambda CDM cosmology using CAMB.

    The cosmological parameters are:
    - Hubble constant (H0): 74 km/s/Mpc
    - Baryon density (ombh2): 0.022
    - Cold dark matter density (omch2): 0.122
    - Neutrino mass sum (mnu): 0.06 eV
    - Curvature (omk): 0
    - Optical depth to reionization (tau): 0.06
    - Scalar amplitude (As): 2e-9
    - Scalar spectral index (ns): 0.965

    The function computes C_l^TT in units of muK^2 for multipole moments
    l from 2 to 3000. The results are saved in a CSV file.
    """
    # Initialize CAMB parameters
    pars = camb.CAMBparams()

    # Set cosmological parameters
    H0 = 74.0  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Physical baryon density
    omch2 = 0.122  # Physical cold dark matter density
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature parameter
    tau = 0.06  # Optical depth to reionization
    
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)

    # Set initial power spectrum parameters
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    pars.InitPower.set_params(As=As, ns=ns)

    # Set maximum multipole moment for calculation
    lmax_calc = 3000
    pars.set_for_lmax(lmax=lmax_calc, lens_potential_accuracy=0)  # lens_potential_accuracy=0 for unlensed

    # We want scalar power spectra
    pars.WantScalars = True
    
    # By default, DoLensing is False, so we get unlensed C_ls from 'total'
    # if lens_potential_accuracy=0.

    # Get results
    results = camb.get_results(pars)

    # Get CMB power spectra
    # CMB_unit='muK' gives C_l in muK^2 when raw_cl=True
    # raw_cl=True gives C_l instead of D_l = l(l+1)C_l/(2pi)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)

    # Extract total C_l^TT (unlensed scalar, as lensing is off by default)
    # powers['total'] has shape (lmax_calc+1, num_spectra)
    # First column (index 0) is C_l^TT
    cl_tt_all = powers['total'][:, 0]  # Units: muK^2

    # We need l from 2 to 3000
    l_min = 2
    l_max = 3000
    
    # Create array of multipole moments l
    ls = np.arange(l_min, l_max + 1)  # l values from 2 to 3000

    # Extract C_l^TT for the desired range of l
    # cl_tt_all is indexed from l=0. So, cl_tt_all[i] corresponds to l=i.
    cl_tt_selected = cl_tt_all[l_min : l_max + 1]  # Units: muK^2

    # Create pandas DataFrame
    df = pd.DataFrame({'l': ls, 'TT': cl_tt_selected})
    df['l'] = df['l'].astype(int)

    # Define file path and ensure directory exists
    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_path = os.path.join(output_dir, 'result.csv')

    # Save to CSV
    df.to_csv(file_path, index=False)

    print("CMB temperature power spectrum calculation complete.")
    print("Results saved to: " + file_path)
    print("\nDataFrame Information:")
    # df.info() prints to stdout, capture it for clean printing
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    print(info_str)
    
    print("\nFirst 5 rows of the data:")
    print(df.head().to_string())
    print("\nLast 5 rows of the data:")
    print(df.tail().to_string())
    print("\nShape of the data (rows, columns):")
    print(df.shape)


if __name__ == '__main__':
    calculate_cmb_power_spectrum()