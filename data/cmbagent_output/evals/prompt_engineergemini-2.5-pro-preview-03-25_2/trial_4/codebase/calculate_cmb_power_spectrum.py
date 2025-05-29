# filename: codebase/calculate_cmb_power_spectrum.py
import os
import camb
import numpy as np
import pandas as pd

def calculate_cmb_power_spectrum():
    r"""
    Calculates the raw CMB temperature power spectrum (unlensed scalar C_l^TT)
    for a flat Lambda CDM cosmology using specified parameters with CAMB.
    The results are saved to a CSV file.
    """
    # Define the path for saving data
    database_path = "data/"
    if not os.path.exists(database_path):
        os.makedirs(database_path)

    # Cosmological parameters
    H0 = 70.0  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Baryon density Omega_b * h^2
    omch2 = 0.122  # Cold dark matter density Omega_c * h^2
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature Omega_k
    tau = 0.06  # Optical depth to reionization
    
    # Initial power spectrum parameters
    As = 2.0e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    
    # Calculation settings
    lmax_calc = 3000  # Maximum multipole moment

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    
    # We want unlensed raw C_ls, so DoLensing=False
    pars.DoLensing = False
    pars.WantScalars = True  # Ensure scalar modes are computed
    pars.WantTensors = False  # Ensure tensor modes are not computed (default r=0)

    # Set lmax for calculation, lens_potential_accuracy is irrelevant if DoLensing=False
    pars.set_for_lmax(lmax_calc, lens_potential_accuracy=0)

    # Suppress CAMB's informational messages
    camb.set_feedback_level(level=0)

    # Get results
    results = camb.get_results(pars)

    # Get CMB power spectra C_l in muK^2
    # 'total' spectra with DoLensing=False gives unlensed scalar spectra.
    # raw_cl=True returns C_l instead of l(l+1)C_l/2pi
    # CMB_unit='muK' returns C_l in muK^2
    powers = results.get_cmb_power_spectra(CMB_unit='muK', raw_cl=True, lmax=lmax_calc, spectra=['total'])
    
    # powers['total'] is an array with columns: l, TT, EE, BB, TE
    # We need the TT spectrum (index 0)
    # The array is indexed by l, from l=0 to lmax_calc.
    all_cltt = powers['total'][:, 0]  # Units: muK^2

    # We need l from 2 to lmax_calc
    l_values = np.arange(2, lmax_calc + 1, dtype=int)
    
    # Extract C_l^TT for l from 2 to lmax_calc
    # all_cltt[0] is C_0^TT, all_cltt[1] is C_1^TT, etc.
    cltt_values = all_cltt[2:lmax_calc + 1]

    # Create a pandas DataFrame
    df = pd.DataFrame({'l': l_values, 'TT': cltt_values})

    # Save to CSV
    output_filename = os.path.join(database_path, "result.csv")
    df.to_csv(output_filename, index=False)

    print("CMB temperature power spectrum calculation complete.")
    print("Results saved to: " + output_filename)
    print("\nFirst 5 rows of the data:")
    print(df.head().to_string())
    print("\nLast 5 rows of the data:")
    print(df.tail().to_string())
    print("\nSummary statistics of the TT power spectrum:")
    # Suppress scientific notation for describe if it makes it less readable
    pd.options.display.float_format = lambda x: '%.4e' % x
    print(df['TT'].describe().to_string())


if __name__ == '__main__':
    calculate_cmb_power_spectrum()