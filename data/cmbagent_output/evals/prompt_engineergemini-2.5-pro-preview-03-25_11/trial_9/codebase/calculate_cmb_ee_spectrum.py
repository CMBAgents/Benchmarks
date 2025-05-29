# filename: codebase/calculate_cmb_ee_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def calculate_cmb_ee_spectrum():
    r"""
    Calculates the CMB E-mode polarization power spectrum l(l+1)C_l^{EE}/(2pi)
    for a flat Lambda CDM cosmology using specified parameters with CAMB.

    The results are saved to a CSV file and key information is printed to the console.
    """
    print("Starting CMB E-mode polarization power spectrum calculation...")

    # Cosmological Parameters
    H0 = 67.5  # Hubble constant (km/s/Mpc)
    ombh2 = 0.022  # Baryon density
    omch2 = 0.122  # Cold dark matter density
    tau = 0.1  # Optical depth to reionization
    ns = 0.95  # Scalar spectral index
    As_factor = 1.8e-9
    As = As_factor * np.exp(2 * tau)  # Scalar amplitude

    lmax_request = 100  # Maximum multipole moment to calculate up to

    print("\nCosmological parameters used:")
    print("H0: " + str(H0) + " km/s/Mpc")
    print("ombh2: " + str(ombh2))
    print("omch2: " + str(omch2))
    print("tau: " + str(tau))
    print("ns: " + str(ns))
    print("As: " + str(As) + " (calculated as " + str(As_factor) + " * exp(2*tau))")
    print("Reionization Model: Exponential, Exponent Power: 2.0")
    print("Calculating spectrum up to l_max = " + str(lmax_request))
    
    # Initialize CAMB parameters
    pars = camb.CAMBparams()

    # Set cosmological parameters
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0, tau=tau)

    # Set initial power spectrum parameters
    pars.InitPower.As = As
    pars.InitPower.ns = ns

    # Set reionization model
    # CAMB's set_reionization function:
    # model: 'Tanh', 'Polynomial', 'Spline', 'HalfGaussian', 'Exponential', 'UserSupplied'
    # exponent: exponent for 'Exponential' and 'HalfGaussian' models
    # use_optical_depth=True is default if optical_depth is provided.
    pars.set_reionization(model='Exponential', exponent=2.0, optical_depth=tau)
    
    # We want scalar spectra, no tensors
    pars.WantTensors = False
    
    # We want unlensed spectra, so DoLensing should be False (default)
    # pars.DoLensing = False # This is the default

    # Set maximum l for calculation accuracy
    # lens_potential_accuracy=0 means no lensing calculation, consistent with DoLensing=False
    pars.set_for_lmax(lmax_request, lens_potential_accuracy=0)

    # Get results
    print("\nRunning CAMB...")
    results = camb.get_results(pars)

    # Get CMB power spectra
    # CMB_unit='muK' gives l(l+1)C_l/(2pi) in muK^2
    # Specify lmax for the output spectra array
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax_request)

    # powers is a dictionary. For unlensed scalar spectra:
    # powers['unlensed_scalar'] is an array (lmax+1, 4) with columns TT, EE, BB, TE
    # We need the EE spectrum (index 1)
    # The array is indexed by l, so cls[l, spectrum_index]
    cls_unlensed_scalar = powers['unlensed_scalar']
    
    # We need l from 2 to lmax_request (inclusive)
    ls = np.arange(2, lmax_request + 1) # Multipole moments from 2 to 100
    
    # Extract D_l^EE = l(l+1)C_l^EE/(2pi)
    # For l=0 and l=1, polarization spectra are zero or not well-defined.
    # cls_unlensed_scalar[l, 1] gives D_l^EE for multipole l.
    EE_spectrum = cls_unlensed_scalar[ls, 1] # Units: muK^2

    # Create pandas DataFrame
    df_results = pd.DataFrame({
        'l': ls,
        'EE': EE_spectrum
    })

    # Ensure 'l' column is integer type
    df_results['l'] = df_results['l'].astype(int)

    # Save results to CSV
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'result.csv')
    
    df_results.to_csv(output_file, index=False)
    print("\nCalculation complete.")
    print("E-mode polarization power spectrum saved to: " + str(output_file))

    # Print head of the DataFrame
    print("\nFirst 5 rows of the result (l, EE [muK^2]):")
    # Set pandas display options for better console output if needed
    pd.set_option('display.width', 100) # Adjust width for console
    pd.set_option('display.max_columns', None) # Show all columns
    print(df_results.head())
    
    # Print last 5 rows of the DataFrame
    print("\nLast 5 rows of the result (l, EE [muK^2]):")
    print(df_results.tail())
    

if __name__ == '__main__':
    calculate_cmb_ee_spectrum()