# filename: codebase/compute_cmb_tt_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_tt_spectrum():
    r"""
    Compute the lensed CMB temperature power spectrum (TT) for a non-flat Lambda CDM cosmology
    using CAMB, with the following parameters:
        - Hubble constant (H0): 67.3 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Neutrino mass sum (mnu): 0.06 eV
        - Curvature (omk): 0.05
        - Optical depth to reionization (tau): 0.06
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965

    The function computes the lensed temperature power spectrum (Dl = l(l+1)Cl/(2pi)) in units of muK^2
    for multipole moments l=2 to l=3000, and saves the results in a CSV file 'result.csv' in the 'data/' directory.

    Returns
    -------
    None
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set cosmological parameters and calculation settings
    pars = camb.set_params(
        H0=67.3,                # Hubble constant [km/s/Mpc]
        ombh2=0.022,            # Baryon density [dimensionless]
        omch2=0.122,            # Cold dark matter density [dimensionless]
        omk=0.05,               # Curvature [dimensionless]
        mnu=0.06,               # Neutrino mass sum [eV]
        tau=0.06,               # Optical depth to reionization [dimensionless]
        As=2e-9,                # Scalar amplitude [dimensionless]
        ns=0.965,               # Scalar spectral index [dimensionless]
        lmax=3000,              # Maximum multipole moment for calculation
        lens_potential_accuracy=1, # Enable lensing with Planck-level accuracy
        WantTensors=False       # Only scalar modes
    )

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Get the lensed scalar CMB power spectra (Dl = l(l+1)Cl/2pi, in muK^2)
    lmax = 3000
    lensed_cls = results.get_lensed_scalar_cls(CMB_unit='muK', lmax=lmax, raw_cl=False)
    # lensed_cls shape: (lmax+1, 4), columns: TT, EE, BB, TE

    # Prepare data for l=2 to l=3000
    ls = np.arange(2, lmax + 1)  # Multipole moments [dimensionless]
    TT = lensed_cls[2:lmax + 1, 0]  # TT spectrum [muK^2]

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'l': ls, 'TT': TT})
    df.to_csv(output_path, index=False, header=['l', 'TT'])

    # Print summary to console
    print("CMB temperature power spectrum (lensed, TT) saved to " + output_path)
    print("Columns: l (multipole, dimensionless), TT (Dl = l(l+1)Cl/(2pi), muK^2)")
    print("First 5 rows:")
    print(df.head(5).to_string(index=False))
    print("Last 5 rows:")
    print(df.tail(5).to_string(index=False))
    print("Total rows: " + str(df.shape[0]))


if __name__ == "__main__":
    compute_cmb_tt_spectrum()