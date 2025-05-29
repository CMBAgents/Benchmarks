# filename: codebase/cmb_tt_spectrum.py
import camb
import numpy as np
import os

def ensure_data_dir_exists():
    r"""
    Ensures that the 'data' directory exists in the current working directory.
    If it does not exist, it is created.

    Returns
    -------
    str
        The path to the 'data' directory.
    """
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir

def compute_cmb_tt_spectrum(
    H0=74.0,
    ombh2=0.022,
    omch2=0.122,
    mnu=0.06,
    omk=0.0,
    tau=0.06,
    As=2.0e-9,
    ns=0.965,
    lmax=3000
):
    r"""
    Computes the raw CMB temperature power spectrum (C_l^{TT}) for a flat Lambda CDM cosmology
    using CAMB, with the specified cosmological parameters.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density, Omega_b h^2.
    omch2 : float
        Physical cold dark matter density, Omega_c h^2.
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature density parameter, Omega_k.
    tau : float
        Optical depth to reionization.
    As : float
        Scalar amplitude of primordial power spectrum.
    ns : float
        Scalar spectral index.
    lmax : int
        Maximum multipole moment l to compute.

    Returns
    -------
    l_values : numpy.ndarray
        Array of multipole moments (l), from 2 to lmax (inclusive).
    TT_values : numpy.ndarray
        Array of raw temperature power spectrum values (C_l^{TT}) in micro-Kelvin squared (muK^2).
    """
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
        WantScalars=True,
        WantTensors=False,
        WantVectors=False
    )

    # Run CAMB calculations
    results = camb.get_results(pars)

    # Get the raw (unlensed) scalar CMB power spectra in muK^2 units
    powers = results.get_cmb_power_spectra(
        pars,
        CMB_unit='muK',
        raw_cl=True,
        spectra=['unlensed_scalar']
    )

    # Extract TT spectrum (column 0) for l=0..lmax
    cl_unlensed_scalar = powers['unlensed_scalar']
    cl_TT_raw_muK2 = cl_unlensed_scalar[:, 0]

    # Prepare l and TT arrays for l=2..lmax
    l_values = np.arange(2, lmax + 1)
    TT_values = cl_TT_raw_muK2[2:lmax + 1]

    return l_values, TT_values

def save_tt_spectrum_to_csv(l_values, TT_values, filename):
    r"""
    Saves the multipole moments and corresponding TT power spectrum values to a CSV file.

    Parameters
    ----------
    l_values : numpy.ndarray
        Array of multipole moments (l).
    TT_values : numpy.ndarray
        Array of TT power spectrum values (C_l^{TT}) in muK^2.
    filename : str
        Path to the output CSV file.

    Returns
    -------
    None
    """
    output_data = np.column_stack((l_values, TT_values))
    header_str = "l,TT"
    np.savetxt(filename, output_data, delimiter=",", header=header_str, fmt=["%d", "%.18e"], comments="")
    print("Raw CMB TT power spectrum (l=2 to l=3000) saved to " + filename)
    print("First 5 rows:")
    for i in range(min(5, len(l_values))):
        print("l = " + str(l_values[i]) + ", TT = " + str(TT_values[i]) + " muK^2")
    print("Total number of rows: " + str(len(l_values)))


if __name__ == "__main__":
    data_dir = ensure_data_dir_exists()
    output_csv = os.path.join(data_dir, "result.csv")
    l_vals, TT_vals = compute_cmb_tt_spectrum()
    save_tt_spectrum_to_csv(l_vals, TT_vals, output_csv)