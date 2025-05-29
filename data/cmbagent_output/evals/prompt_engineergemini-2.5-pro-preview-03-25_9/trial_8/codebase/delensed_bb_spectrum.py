# filename: codebase/delensed_bb_spectrum.py
import os
import numpy as np
import pandas as pd
import camb

def calculate_delensed_bb_spectrum():
    r"""
    Calculates the delensed CMB B-mode polarization power spectrum (C_ell^BB)
    for a flat Lambda CDM cosmology using CAMB.

    The calculation uses specified cosmological parameters and a delensing
    efficiency of 10%. The results (l, C_ell^BB in muK^2) for l=2 to 3000
    are saved to a CSV file.
    """

    # Cosmological parameters
    H0 = 67.5  # Hubble constant (km/s/Mpc)
    ombh2 = 0.022  # Baryon density * h^2
    omch2 = 0.122  # Cold dark matter density * h^2
    mnu = 0.06  # Sum of neutrino masses (eV)
    omk = 0.0  # Curvature fraction (Omega_k)
    tau = 0.06  # Optical depth to reionization
    r_tensor = 0.1  # Tensor-to-scalar ratio at k_pivot=0.05 Mpc^-1
    As_scalar = 2e-9  # Scalar amplitude at k_pivot=0.05 Mpc^-1
    ns_scalar = 0.965  # Scalar spectral index at k_pivot=0.05 Mpc^-1
    
    delensing_efficiency = 0.10  # 10% delensing efficiency

    lmax_calc = 3000  # Maximum multipole moment for calculation and output

    print("Setting up CAMB parameters...")
    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As_scalar, ns=ns_scalar, r=r_tensor)
    pars.WantTensors = True  # Calculate tensor modes (primordial gravitational waves)
    
    # Set maximum multipole for calculations
    # lens_potential_accuracy=1 is the default, affecting lensed spectra calculation.
    pars.set_for_lmax(lmax=lmax_calc, lens_potential_accuracy=1)

    print("Running CAMB to get power spectra...")
    # Get results from CAMB
    results = camb.get_results(pars)

    # Get CMB power spectra in muK^2 units
    # powers is a dictionary containing 'total', 'unlensed_scalar', 'lensed_scalar', 'tensor' spectra.
    # Each item is an array of shape (4, lmax_calc+1), where 4 corresponds to TT, EE, BB, TE.
    # The second dimension covers multipoles L from 0 to lmax_calc.
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK^2', lmax=lmax_calc)

    # Extract B-mode components (index 2 for BB)
    # C_l_BB_primordial: Primordial B-modes from tensor modes. Array for l = 0, ..., lmax_calc.
    C_l_BB_primordial = powers['tensor'][2, :]
    
    # C_l_BB_lensing: Lensing-induced B-modes from scalar modes. Array for l = 0, ..., lmax_calc.
    C_l_BB_lensing = powers['lensed_scalar'][2, :]

    print("Applying delensing efficiency...")
    # Calculate delensed B-mode power spectrum:
    # C_l^BB(delensed) = C_l^BB(primordial) + (1 - efficiency) * C_l^BB(lensing)
    C_l_BB_delensed = C_l_BB_primordial + (1.0 - delensing_efficiency) * C_l_BB_lensing

    # Prepare data for CSV output
    # Multipole moments from l=2 to l=lmax_calc (inclusive)
    ls = np.arange(2, lmax_calc + 1)  # Array of l values: [2, 3, ..., lmax_calc]
    
    # Select the C_l_BB_delensed values for the range l=2 to lmax_calc.
    # Array indices for l=2 to lmax_calc are also 2 to lmax_calc.
    C_l_BB_delensed_selected = C_l_BB_delensed[2 : lmax_calc + 1]

    print("Preparing data for CSV file...")
    # Create a Pandas DataFrame
    df_results = pd.DataFrame({
        'l': ls.astype(int),
        'BB': C_l_BB_delensed_selected  # This is C_ell^BB in muK^2
    })

    # Create data directory if it doesn't exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("Created directory: " + data_dir)

    # Save to CSV
    file_path = os.path.join(data_dir, 'result.csv')
    df_results.to_csv(file_path, index=False)

    print("Successfully calculated delensed B-mode power spectrum.")
    print("Results saved to: " + file_path)
    
    # Print summary of the results
    print("\nSummary of the results:")
    print("Number of multipoles calculated: " + str(len(df_results)))
    print("Multipole range: l=" + str(df_results['l'].min()) + " to l=" + str(df_results['l'].max()))
    
    print("\nFirst 5 rows of the data:")
    print(df_results.head().to_string()) 
    
    print("\nLast 5 rows of the data:")
    print(df_results.tail().to_string())

    print("\nDescriptive statistics for BB (muK^2):")
    desc_stats = df_results['BB'].describe()
    print("Mean: " + str(desc_stats['mean']))
    print("Std:  " + str(desc_stats['std']))
    print("Min:  " + str(desc_stats['min']))
    print("25%:  " + str(desc_stats['25%']))  # 25th percentile
    print("50%:  " + str(desc_stats['50%']))  # Median (50th percentile)
    print("75%:  " + str(desc_stats['75%']))  # 75th percentile
    print("Max:  " + str(desc_stats['max']))

if __name__ == "__main__":
    calculate_delensed_bb_spectrum()