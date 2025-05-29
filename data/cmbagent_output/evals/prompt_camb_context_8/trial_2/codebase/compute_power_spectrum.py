# filename: codebase/compute_power_spectrum.py
import os
import camb
from camb import model
import numpy as np
import pandas as pd

def compute_linear_matter_power_spectrum():
    r"""
    Computes the linear matter power spectrum P(k) at z=0 for a flat Lambda CDM cosmology
    using CAMB, and saves the results to 'data/result.csv'.

    Returns
    -------
    None
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cosmological parameters (units in comments)
    H0 = 67.5          # Hubble constant [km/s/Mpc]
    ombh2 = 0.022      # Omega_b * h^2 [dimensionless]
    omch2 = 0.122      # Omega_c * h^2 [dimensionless]
    mnu = 0.06         # Sum of neutrino masses [eV]
    omk = 0.0          # Curvature [dimensionless]
    tau = 0.06         # Optical depth [dimensionless]
    As = 2.0e-9        # Scalar amplitude [dimensionless]
    ns = 0.965         # Scalar spectral index [dimensionless]
    kmax_calc = 2.0    # Maximum k for CAMB internal calculation [Mpc^-1]

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
        WantTransfer=True,
        redshifts=[0.0],
        kmax=kmax_calc
    )

    # Ensure linear matter power spectrum is used
    pars.NonLinear = model.NonLinear_none

    # Run CAMB
    results = camb.get_results(pars)

    # Output grid for P(k)
    minkh_output = 1e-4   # Minimum k/h [h/Mpc]
    maxkh_output = 1.0    # Maximum k/h [h/Mpc]
    npoints_output = 200  # Number of k/h points

    # Get linear matter power spectrum at z=0
    kh, z_out, pk_out = results.get_matter_power_spectrum(
        minkh=minkh_output,
        maxkh=maxkh_output,
        npoints=npoints_output
    )
    # kh: array of k/h [h/Mpc]
    # z_out: array of redshifts
    # pk_out: 2D array, shape (len(z_out), len(kh)), units (Mpc/h)^3

    # Extract P(k) at z=0
    if z_out[0] == 0.0:
        pk_at_z0 = pk_out[0, :]
    else:
        raise RuntimeError("z=0 not found in CAMB output.")

    # Save to CSV
    output_file = os.path.join(output_dir, "result.csv")
    df = pd.DataFrame({
        "kh": kh,        # [h/Mpc]
        "P_k": pk_at_z0  # [(Mpc/h)^3]
    })
    df.to_csv(output_file, index=False, float_format="%.8e")

    # Print summary to console
    print("Linear matter power spectrum P(k) at z=0 saved to " + output_file)
    print("kh (h/Mpc) range: " + "%.3e" % (kh.min()) + " to " + "%.3e" % (kh.max()))
    print("P_k ((Mpc/h)^3) range: " + "%.3e" % (pk_at_z0.min()) + " to " + "%.3e" % (pk_at_z0.max()))
    print("Number of k points: " + str(len(kh)))
    print("\nFirst 5 rows of the output:")
    print(df.head())


if __name__ == "__main__":
    compute_linear_matter_power_spectrum()