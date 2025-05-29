# filename: codebase/compute_cmb_ee_spectrum.py
import camb
import numpy as np
import math
import os
import pandas as pd

def compute_cmb_ee_spectrum(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density [dimensionless]
    omch2=0.122,            # Cold dark matter density [dimensionless]
    tau=0.1,                # Optical depth to reionization [dimensionless]
    As_base=1.8e-9,         # Base scalar amplitude [dimensionless]
    ns=0.95,                # Scalar spectral index [dimensionless]
    reion_exp_power=2,      # Exponential reionization exponent [dimensionless]
    lmin=2,                 # Minimum multipole [dimensionless]
    lmax=100,               # Maximum multipole [dimensionless]
    output_csv='data/result.csv' # Output CSV file path
):
    """
    Compute the CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2π)) in units of μK^2
    for a flat Lambda CDM cosmology using CAMB, with exponential reionization (exponent=2).
    Results are saved to a CSV file with columns 'l' and 'EE'.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter (Ω_b h^2).
    omch2 : float
        Physical cold dark matter density parameter (Ω_c h^2).
    tau : float
        Optical depth to reionization.
    As_base : float
        Base scalar amplitude (A_s, before tau correction).
    ns : float
        Scalar spectral index.
    reion_exp_power : float
        Exponent for exponential reionization model.
    lmin : int
        Minimum multipole moment to include in output.
    lmax : int
        Maximum multipole moment to include in output.
    output_csv : str
        Path to output CSV file.

    Returns
    -------
    None
    """
    # Calculate A_s with tau correction [dimensionless]
    As = As_base * math.exp(2.0 * tau)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up CAMB parameters
    pars = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        omk=0.0,  # Flat universe
        tau=tau,
        As=As,
        ns=ns,
        reionization_model='ExpReionization',
        reion_exp_power=reion_exp_power,
        WantScalars=True,
        WantTensors=False,
        lmax=lmax,
        lens_potential_accuracy=1
    )

    # Compute results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in μK^2, D_l = l(l+1)C_l/(2π)
    powers = results.get_cmb_power_spectra(
        pars,
        CMB_unit='muK',
        spectra=['lensed_scalar'],
        lmax=lmax
    )
    # powers['lensed_scalar'] shape: (lmax+1, 4), columns: TT, EE, BB, TE
    Dl_EE = powers['lensed_scalar'][:, 1]  # EE spectrum [μK^2]

    # Prepare l and EE arrays for output (l=2..lmax)
    l_arr = np.arange(lmin, lmax + 1, dtype=int)
    EE_arr = Dl_EE[lmin:lmax + 1]

    # Save to CSV using pandas for clarity
    df = pd.DataFrame({'l': l_arr, 'EE': EE_arr})
    df.to_csv(output_csv, index=False)

    # Print summary to console
    print("CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2π)) [μK^2] saved to " + output_csv)
    print("First 5 rows:")
    print(df.head())
    print("Total rows saved: " + str(len(df)))
    print("l range: " + str(l_arr[0]) + " to " + str(l_arr[-1]))
    print("EE (μK^2) min: " + str(np.min(EE_arr)) + ", max: " + str(np.max(EE_arr)))

if __name__ == "__main__":
    compute_cmb_ee_spectrum()