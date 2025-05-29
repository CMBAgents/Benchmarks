# filename: codebase/compute_cmb_tt.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_tt_spectrum():
    r"""
    Compute the lensed scalar CMB temperature power spectrum (C_l^{TT}) for a flat Lambda CDM cosmology
    using CAMB, for l=2 to l=3000, in units of μK^2.

    Returns:
        l_values (np.ndarray): Array of multipole moments (l), shape (2999,), units: dimensionless
        Cl_TT (np.ndarray): Array of C_l^{TT} values, shape (2999,), units: μK^2
    """
    # Set cosmological parameters
    pars = camb.set_params(
        H0=70.0,           # Hubble constant [km/s/Mpc]
        ombh2=0.022,       # Baryon density Ω_b h^2
        omch2=0.122,       # Cold dark matter density Ω_c h^2
        mnu=0.06,          # Neutrino mass sum [eV]
        omk=0.0,           # Curvature Ω_k (flat universe)
        tau=0.06,          # Optical depth to reionization
        As=2e-9,           # Scalar amplitude
        ns=0.965,          # Scalar spectral index
        lmax=3000,         # Maximum multipole
        WantScalars=True,
        WantTensors=False,
        WantVectors=False
    )

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Get the lensed scalar CMB power spectra (raw Cl, in μK^2)
    powers = results.get_cmb_power_spectra(
        params=pars,
        lmax=3000,
        spectra=['lensed_scalar'],
        CMB_unit='muK',
        raw_cl=True
    )

    # Extract TT spectrum for l=2..3000
    Cl_TT_all = powers['lensed_scalar'][:, 0]  # TT is column 0
    l_values = np.arange(2, 3001)              # l = 2..3000
    Cl_TT = Cl_TT_all[2:3001]                  # C_l^{TT} for l=2..3000

    return l_values, Cl_TT

def save_spectrum_to_csv(l_values, Cl_TT, filename):
    r"""
    Save the CMB TT power spectrum to a CSV file.

    Args:
        l_values (np.ndarray): Array of multipole moments (l), shape (N,), units: dimensionless
        Cl_TT (np.ndarray): Array of C_l^{TT} values, shape (N,), units: μK^2
        filename (str): Output CSV filename
    """
    df = pd.DataFrame({'l': l_values, 'TT': Cl_TT})
    df.to_csv(filename, index=False)
    print("CMB TT power spectrum (l=2..3000) saved to " + filename)
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())
    print("Total rows: " + str(len(df)))


if __name__ == "__main__":
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Compute the spectrum
    l_values, Cl_TT = compute_cmb_tt_spectrum()

    # Save to CSV
    output_csv = os.path.join(output_dir, "result.csv")
    save_spectrum_to_csv(l_values, Cl_TT, output_csv)