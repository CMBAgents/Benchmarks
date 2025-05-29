# filename: codebase/compute_cmb_ee_spectrum.py
import os
import math
import numpy as np
import camb
import pandas as pd

def compute_cmb_ee_spectrum():
    """
    Compute the CMB E-mode polarization power spectrum for the specified flat Lambda CDM cosmology
    and save the results to 'data/result.csv'.

    Returns
    -------
    None
    """
    # Cosmological parameters (units in comments)
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    tau = 0.1  # Optical depth to reionization [dimensionless]
    ns = 0.95  # Scalar spectral index [dimensionless]
    As_base = 1.8e-9  # Scalar amplitude base [dimensionless]
    As = As_base * math.exp(2 * tau)  # Scalar amplitude [dimensionless]
    omk = 0.0  # Flat universe

    # Reionization model parameters
    reionization_model = 'ExpReionization'
    reion_exp_power = 2  # Exponent for exponential reionization [dimensionless]

    # Multipole range
    lmin = 2
    lmax = 100  # Maximum multipole for output [dimensionless]
    lmax_calc = 120  # Slightly higher for calculation accuracy

    # Set up CAMB parameters
    pars = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        omk=omk,
        tau=tau,
        ns=ns,
        As=As,
        reionization_model=reionization_model,
        reion_exp_power=reion_exp_power,
        lmax=lmax_calc,
        lens_potential_accuracy=1,
        WantScalars=True,
        WantTensors=False
    )

    # Run CAMB
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in muK^2, Dl = l(l+1)Cl/(2pi)
    powers = results.get_cmb_power_spectra(
        pars,
        CMB_unit='muK',
        spectra=['lensed_scalar'],
        lmax=lmax
    )
    # powers['lensed_scalar'] shape: (lmax+1, 4), columns: TT, EE, BB, TE
    Dl_EE = powers['lensed_scalar'][:, 1]  # EE column, muK^2

    # Prepare output for l=2..100
    l_arr = np.arange(lmin, lmax + 1, dtype=int)
    EE_arr = Dl_EE[lmin:lmax + 1]

    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({'l': l_arr, 'EE': EE_arr})
    df.to_csv(output_path, index=False)

    # Print summary to console
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print("CMB E-mode polarization power spectrum (l(l+1)C_l^{EE}/(2pi)) [muK^2] for l=2..100")
    print("Saved to: " + output_path)
    print("First 10 rows:")
    print(df.head(10).to_string(index=False))
    print("Last 5 rows:")
    print(df.tail(5).to_string(index=False))
    print("Total rows saved: " + str(len(df)))


if __name__ == "__main__":
    compute_cmb_ee_spectrum()
