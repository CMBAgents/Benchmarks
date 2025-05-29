# filename: codebase/cmb_tt_spectrum.py
import camb
from camb import model
import numpy as np
import os

def compute_cmb_tt_spectrum(
    H0=67.5,           # Hubble constant [km/s/Mpc]
    ombh2=0.02,        # Baryon density Omega_b h^2
    omch2=0.122,       # Cold dark matter density Omega_c h^2
    mnu=0.06,          # Neutrino mass sum [eV]
    omk=0.0,           # Curvature Omega_k (flat)
    tau=0.06,          # Optical depth to reionization
    As=2e-9,           # Scalar amplitude
    ns=0.965,          # Scalar spectral index
    lmax=3000,         # Maximum multipole
    output_dir="data", # Output directory
    output_filename="result.csv" # Output CSV filename
):
    r"""
    Compute the CMB temperature power spectrum l(l+1)C_l^{TT}/(2pi) in units of microKelvin^2
    for a flat Lambda CDM cosmology using CAMB, and save the results to a CSV file.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter Omega_b h^2.
    omch2 : float
        Physical cold dark matter density parameter Omega_c h^2.
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter Omega_k (0 for flat cosmology).
    tau : float
        Optical depth to reionization.
    As : float
        Scalar amplitude of primordial power spectrum.
    ns : float
        Scalar spectral index.
    lmax : int
        Maximum multipole moment l to compute.
    output_dir : str
        Directory to save the output CSV file.
    output_filename : str
        Name of the output CSV file.

    Returns
    -------
    None
        The function saves the computed spectrum to a CSV file and prints a summary.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
    # Use non-linear corrections for lensing and matter power spectrum
    pars.NonLinear = model.NonLinear_both
    pars.WantCls = True
    pars.Want_CMB = True
    pars.DoLensing = True

    # Run CAMB and get results
    results = camb.get_results(pars)

    # Get lensed CMB power spectra in microKelvin^2
    powers = results.get_cmb_power_spectra(CMB_unit='muK')
    total_cls = powers['total']  # shape: (lmax+1, 4), columns: TT, EE, BB, TE

    # Extract TT spectrum for l=2 to lmax
    l_values = np.arange(2, lmax + 1)  # Multipole moments (dimensionless)
    tt_values = total_cls[2:lmax + 1, 0]  # TT spectrum in microKelvin^2

    # Save to CSV
    output_path = os.path.join(output_dir, output_filename)
    data_to_save = np.column_stack((l_values, tt_values))
    np.savetxt(output_path, data_to_save, delimiter=',', header='l,TT', comments='')

    # Print summary
    print("CMB TT power spectrum calculation complete.")
    print("Results saved to " + output_path)
    print("First 5 rows (l, TT [microKelvin^2]):")
    for i in range(min(5, len(l_values))):
        print("l = " + str(int(l_values[i])) + ", TT = " + str(tt_values[i]))

if __name__ == "__main__":
    compute_cmb_tt_spectrum()