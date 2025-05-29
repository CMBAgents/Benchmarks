# filename: codebase/compute_cmb_tt_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_tt_spectrum():
    r"""
    Compute the raw (unlensed) CMB temperature power spectrum (C_l^{TT}) for a flat Lambda CDM cosmology
    using CAMB, with the following parameters:
        - Hubble constant (H0): 70 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Neutrino mass sum (mnu): 0.06 eV
        - Curvature (omk): 0
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965
    The spectrum is computed for multipole moments l=2 to l=3000, in units of microkelvin squared (muK^2).
    The results are saved in a CSV file 'result.csv' in the 'data/' directory, with columns:
        - l: Multipole moment (integer, 2 to 3000)
        - TT: Temperature power spectrum (C_l^{TT} in muK^2)
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set cosmological parameters
    params = camb.set_params(
        H0=70.0,           # Hubble constant [km/s/Mpc]
        ombh2=0.022,       # Baryon density Omega_b h^2
        omch2=0.122,       # Cold dark matter density Omega_c h^2
        mnu=0.06,          # Neutrino mass sum [eV]
        omk=0.0,           # Curvature Omega_k
        tau=0.06,          # Optical depth to reionization
        As=2.0e-9,         # Scalar amplitude
        ns=0.965,          # Scalar spectral index
        lmax=3000          # Maximum multipole
    )
    
    # Run CAMB calculation
    results = camb.get_results(params)
    
    # Get the raw (unlensed) CMB TT power spectrum in muK^2
    cl_dict = results.get_cmb_power_spectra(
        params=params,
        CMB_unit='muK',
        raw_cl=True,
        spectra=['unlensed_scalar']
    )
    cl_TT = cl_dict['unlensed_scalar'][:, 0]  # TT column, units: muK^2
    
    # Prepare l and TT arrays for l=2 to 3000
    l_vals = np.arange(2, 3001)  # Multipole moments (dimensionless)
    TT_vals = cl_TT[2:3001]      # C_l^{TT} [muK^2]
    
    # Save to CSV
    output_df = pd.DataFrame({'l': l_vals, 'TT': TT_vals})
    output_path = os.path.join(output_dir, "result.csv")
    output_df.to_csv(output_path, index=False)
    
    # Print summary to console
    print("CMB temperature power spectrum (C_l^{TT}) computed for l=2 to l=3000.")
    print("Units: TT in microkelvin squared (muK^2).")
    print("Results saved to " + output_path)
    print("First 5 rows of the output:")
    print(output_df.head())

if __name__ == "__main__":
    compute_cmb_tt_spectrum()