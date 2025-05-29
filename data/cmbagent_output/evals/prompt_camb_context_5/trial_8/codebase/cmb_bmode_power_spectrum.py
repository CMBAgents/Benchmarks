# filename: codebase/cmb_bmode_power_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def compute_cmb_bmode_power_spectrum(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density parameter
    omch2=0.122,            # Cold dark matter density parameter
    mnu=0.06,               # Neutrino mass sum [eV]
    omk=0.0,                # Curvature parameter
    tau=0.06,               # Optical depth to reionization
    As=2e-9,                # Scalar amplitude
    ns=0.965,               # Scalar spectral index
    r=0.0,                  # Tensor-to-scalar ratio
    lmax=3000,              # Maximum multipole
    lens_potential_accuracy=1, # Lensing accuracy
    output_csv='data/result.csv' # Output CSV file path
):
    r"""
    Compute the CMB B-mode polarization power spectrum D_l^{BB} = l(l+1)C_l^{BB}/(2pi)
    for a flat Lambda CDM cosmology using CAMB, with the specified parameters.

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.
    ombh2 : float
        Physical baryon density parameter (Omega_b h^2).
    omch2 : float
        Physical cold dark matter density parameter (Omega_c h^2).
    mnu : float
        Sum of neutrino masses in eV.
    omk : float
        Curvature parameter (Omega_k).
    tau : float
        Optical depth to reionization.
    As : float
        Scalar amplitude.
    ns : float
        Scalar spectral index.
    r : float
        Tensor-to-scalar ratio.
    lmax : int
        Maximum multipole moment to compute.
    lens_potential_accuracy : int
        Accuracy parameter for lensing calculation.
    output_csv : str
        Path to output CSV file.

    Returns
    -------
    None
        Saves the B-mode power spectrum to a CSV file with columns 'l' and 'BB'.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set cosmological parameters
    pars = camb.set_params(
        H0=H0,
        ombh2=ombh2,
        omch2=omch2,
        mnu=mnu,
        omk=omk,
        tau=tau,
        As=As,
        ns=ns,
        r=r,
        lmax=lmax,
        lens_potential_accuracy=lens_potential_accuracy
    )

    # Run CAMB to get results
    results = camb.get_results(pars)

    # Get lensed scalar CMB power spectra in muK^2 units
    # Output: array of shape (lmax+1, 4), columns: TT, EE, BB, TE
    powers = results.get_lensed_scalar_cls(lmax=lmax, CMB_unit='muK')

    # Multipole moments (l = 0 to lmax)
    ls = np.arange(powers.shape[0])

    # Extract B-mode power spectrum (column 2)
    dl_bb = powers[:, 2]  # units: muK^2

    # Select l = 2 to lmax (inclusive)
    lmin = 2
    lmax_out = lmax
    ls_out = ls[lmin:lmax_out+1]
    dl_bb_out = dl_bb[lmin:lmax_out+1]

    # Save to CSV
    df = pd.DataFrame({'l': ls_out, 'BB': dl_bb_out})
    df.to_csv(output_csv, index=False)

    # Print summary to console
    print("CMB B-mode polarization power spectrum (lensed, scalar, muK^2) saved to " + output_csv)
    print("Columns: l (multipole moment), BB (D_l^{BB} in muK^2)")
    print("First 5 rows:")
    print(df.head())
    print("Last 5 rows:")
    print(df.tail())
    print("Total rows: " + str(len(df)))
    print("l range: " + str(ls_out[0]) + " to " + str(ls_out[-1]))
    print("BB min: " + str(np.min(dl_bb_out)) + " muK^2, BB max: " + str(np.max(dl_bb_out)) + " muK^2")


if __name__ == "__main__":
    compute_cmb_bmode_power_spectrum()