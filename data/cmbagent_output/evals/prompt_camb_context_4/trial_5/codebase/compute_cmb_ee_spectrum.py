# filename: codebase/compute_cmb_ee_spectrum.py
import camb
import numpy as np
import pandas as pd
import os


def compute_cmb_ee_spectrum():
    r"""
    Compute the CMB E-mode polarization power spectrum D_ell^{EE} = ell(ell+1)C_ell^{EE}/(2pi)
    in units of microkelvin squared (muK^2) for a flat Lambda CDM cosmology using CAMB.

    Cosmological parameters:
        H0: Hubble constant [km/s/Mpc]
        ombh2: Physical baryon density [dimensionless]
        omch2: Physical cold dark matter density [dimensionless]
        mnu: Sum of neutrino masses [eV]
        omk: Curvature density parameter [dimensionless]
        tau: Optical depth to reionization [dimensionless]
        As: Scalar amplitude [dimensionless]
        ns: Scalar spectral index [dimensionless]

    The function computes the lensed scalar CMB power spectra up to l=3000,
    extracts the E-mode (EE) spectrum, and saves the results to a CSV file.

    Output:
        data/result.csv: CSV file with columns:
            l: Multipole moment (integer, 2..3000)
            EE: E-mode polarization power spectrum D_ell^{EE} [muK^2]
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density [dimensionless]
    omch2 = 0.122  # Cold dark matter density [dimensionless]
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature [dimensionless]
    tau = 0.04  # Optical depth [dimensionless]
    As = 2.0e-9  # Scalar amplitude [dimensionless]
    ns = 0.965  # Scalar spectral index [dimensionless]
    l_max_scalar = 3000  # Maximum multipole

    # Set up CAMB parameters
    pars = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu,
        omk=omk,
        tau=tau,
        As=As,
        ns=ns,
        WantScalars=True,
        WantTensors=False,
        DoLensing=True
    )
    # Set lmax and lensing accuracy
    pars.set_for_lmax(l_max_scalar, lens_potential_accuracy=1)

    # Run CAMB
    results = camb.get_results(pars)

    # Extract lensed scalar CMB power spectra in muK^2
    powers = results.get_cmb_power_spectra(CMB_unit='muK', spectra=['lensed_scalar'], lmax=l_max_scalar)
    lensed_scalar_cls = powers['lensed_scalar']  # shape: (lmax+1, 4), columns: TT, EE, BB, TE

    # Multipole moments (l) from 2 to l_max_scalar
    ell = np.arange(2, l_max_scalar + 1)
    # E-mode polarization power spectrum [muK^2]
    EE_spectrum = lensed_scalar_cls[2:l_max_scalar + 1, 1]

    # Prepare output directory
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save to CSV
    output_filename = os.path.join(output_dir, "result.csv")
    df_results = pd.DataFrame({
        'l': ell,
        'EE': EE_spectrum
    })
    df_results.to_csv(output_filename, index=False)

    # Print summary to console
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    print("CMB E-mode polarization power spectrum (D_ell^{EE} = l(l+1)C_ell^{EE}/(2pi)) [muK^2] saved to " + output_filename)
    print("First 5 rows:")
    print(df_results.head(5).to_string(index=False))
    print("Last 5 rows:")
    print(df_results.tail(5).to_string(index=False))
    print("Total number of multipoles: " + str(len(df_results)))


if __name__ == "__main__":
    compute_cmb_ee_spectrum()
