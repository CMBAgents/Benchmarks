# filename: codebase/compute_cmb_ee_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_ee_spectrum(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density Omega_b h^2
    omch2=0.122,            # Cold dark matter density Omega_c h^2
    mnu=0.06,               # Sum of neutrino masses [eV]
    omk=0.0,                # Curvature Omega_k (0 for flat)
    tau=0.04,               # Optical depth to reionization
    As=2e-9,                # Scalar amplitude
    ns=0.965,               # Scalar spectral index
    lmax=3000,              # Maximum multipole
    output_folder="data/",  # Output directory
    output_filename="result.csv" # Output CSV filename
):
    r"""
    Compute the CMB E-mode polarization power spectrum D_l^{EE} = l(l+1)C_l^{EE}/(2pi)
    in units of microKelvin^2 for a flat Lambda CDM cosmology using CAMB.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Baryon density Omega_b h^2.
    omch2 : float
        Cold dark matter density Omega_c h^2.
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature Omega_k (0 for flat).
    tau : float
        Optical depth to reionization.
    As : float
        Scalar amplitude.
    ns : float
        Scalar spectral index.
    lmax : int
        Maximum multipole to compute.
    output_folder : str
        Directory to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
        Saves the results to a CSV file with columns 'l' and 'EE'.
    """
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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
        lmax=lmax,
        lens_potential_accuracy=1,
        WantScalars=True,
        WantTensors=False,
        DoLensing=True
    )

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in muK^2, D_ell = l(l+1)C_ell/(2pi)
    powers = results.get_cmb_power_spectra(
        params=pars,
        lmax=lmax,
        spectra=['lensed_scalar'],
        CMB_unit='muK',
        raw_cl=False
    )

    # Extract EE spectrum (column 1)
    lensed_scalar_cls = powers['lensed_scalar']
    cl_EE = lensed_scalar_cls[:, 1]  # EE column

    # Prepare l and EE arrays for l=2 to lmax
    ls = np.arange(2, lmax + 1)
    cl_EE_selected = cl_EE[2:lmax + 1]

    # Save to CSV
    output_path = os.path.join(output_folder, output_filename)
    df = pd.DataFrame({'l': ls, 'EE': cl_EE_selected})
    df.to_csv(output_path, index=False)

    # Print summary of results
    np.set_printoptions(precision=6, suppress=True)
    print("E-mode polarization power spectrum (D_l^EE = l(l+1)C_l^EE/(2pi)) saved to " + output_path)
    print("Units: l (dimensionless), EE (microKelvin^2)")
    print("\nFirst 5 rows:")
    print(df.head(5).to_string(index=False))
    print("\nLast 5 rows:")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    compute_cmb_ee_spectrum()