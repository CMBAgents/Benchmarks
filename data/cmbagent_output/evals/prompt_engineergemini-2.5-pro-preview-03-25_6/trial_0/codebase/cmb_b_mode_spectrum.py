# filename: codebase/cmb_b_mode_spectrum.py
import os
import numpy as np
import pandas as pd
import camb

def calculate_cmb_b_mode_spectrum():
    r"""
    Calculates the CMB raw B-mode polarization power spectrum (C_l^BB)
    for a flat Lambda CDM cosmology using CAMB and saves it to a CSV file.

    The function uses a specific set of cosmological parameters:
    - Hubble constant (H0): 67.5 km/s/Mpc
    - Baryon density (ombh2): 0.022
    - Cold dark matter density (omch2): 0.122
    - Neutrino mass sum (mnu): 0.06 eV
    - Curvature (omk): 0
    - Optical depth to reionization (tau): 0.06
    - Tensor-to-scalar ratio (r): 0.1
    - Scalar amplitude (As): 2e-9
    - Scalar spectral index (ns): 0.965

    The B-mode power spectrum (C_l^BB) is computed in units of muK^2
    for multipole moments from l=2 to l=3000.
    The results are saved in 'data/result.csv'.
    """

    # Cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Physical baryon density
    omch2 = 0.122  # Physical cold dark matter density
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature density
    tau = 0.06  # Optical depth to reionization

    # Initial power spectrum parameters
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    r_tensor_to_scalar = 0.1  # Tensor-to-scalar ratio

    # CAMB setup
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=r_tensor_to_scalar)
    
    # We want tensor modes for primordial B-modes
    pars.WantTensors = True

    # Set lmax for calculation and output
    # Calculate up to a slightly higher lmax for better accuracy at the edge
    lmax_calc = 3050 
    lmax_out = 3000  # Output lmax as requested

    # lens_potential_accuracy=0 as we want raw primordial (unlensed) B-modes
    # from tensor perturbations.
    pars.set_for_lmax(lmax_calc, lens_potential_accuracy=0)

    # Get results
    results = camb.get_results(pars)

    # Get CMB power spectra
    # CMB_output_scale=camb.constants.TCMB_uk**2 scales C_l to muK^2
    # raw_cl=True returns C_l instead of l(l+1)C_l/2pi
    # spectra=['tensor'] gives only tensor mode contributions
    powers = results.get_cmb_power_spectra(pars, lmax=lmax_out, 
                                           spectra=['tensor'], 
                                           CMB_output_scale=camb.constants.TCMB_uk**2, 
                                           raw_cl=True)
    
    # The 'tensor' field contains an array with columns: TT, EE, BB, TE
    # We need the BB spectrum, which is the 3rd column (index 2)
    # The array is indexed by l, from l=0 to lmax_out.
    # cl_bb[l] is C_l^BB in muK^2
    tensor_cls = powers['tensor']
    cl_bb = tensor_cls[:, 2]  # Units: muK^2

    # Prepare data for CSV
    # Multipole moments from l=2 to l=3000
    l_values = np.arange(2, lmax_out + 1)  # l from 2 to 3000
    
    # Corresponding C_l^BB values
    # cl_bb is 0-indexed, so cl_bb[2] is C_2^BB, cl_bb[lmax_out] is C_{lmax_out}^BB
    bb_values = cl_bb[2 : lmax_out + 1]  # Slice from l=2 up to l=lmax_out

    df = pd.DataFrame({'l': l_values, 'BB': bb_values})

    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save to CSV
    output_filename = "result.csv"
    output_path = os.path.join(data_dir, output_filename)
    df.to_csv(output_path, index=False)

    print("CMB B-mode power spectrum calculation complete.")
    print("Results saved to: " + str(output_path))
    
    # Print summary of the data
    print("\nFirst 5 rows of the B-mode power spectrum data (l, C_l^BB [muK^2]):")
    print(df.head())
    print("\nLast 5 rows of the B-mode power spectrum data (l, C_l^BB [muK^2]):")
    print(df.tail())

if __name__ == '__main__':
    calculate_cmb_b_mode_spectrum()