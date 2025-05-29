# filename: codebase/compute_linear_power_spectrum.py
import camb
from camb import model
import numpy as np
import pandas as pd
import os

def compute_linear_matter_power_spectrum():
    r"""
    Compute the linear matter power spectrum P(k) at z=0 for a flat Lambda CDM cosmology
    using CAMB with specified parameters. The result is saved as a CSV file with columns:
    'kh' (wavenumber in h/Mpc) and 'P_k' (power spectrum in (Mpc/h)^3).

    Cosmological parameters:
        Hubble constant (H0): 67.5 km/s/Mpc
        Baryon density (ombh2): 0.022
        Cold dark matter density (omch2): 0.122
        Neutrino mass sum (mnu): 0.06 eV
        Curvature (omk): 0
        Optical depth to reionization (tau): 0.06
        Scalar amplitude (As): 2e-9
        Scalar spectral index (ns): 0.965
        k maximum (kmax): 2 (Mpc^-1)

    Output:
        CSV file 'data/result.csv' with columns:
            kh: Wavenumber (in h/Mpc, 200 evenly spaced values)
            P_k: Linear matter power spectrum (in (Mpc/h)^3)
    """
    # Ensure output directory exists
    output_dir = "data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cosmological parameters (units in comments)
    H0 = 67.5            # Hubble constant [km/s/Mpc]
    ombh2 = 0.022        # Omega_b * h^2 [dimensionless]
    omch2 = 0.122        # Omega_c * h^2 [dimensionless]
    mnu = 0.06           # Sum of neutrino masses [eV]
    omk = 0.0            # Curvature [dimensionless]
    tau = 0.06           # Optical depth [dimensionless]
    As = 2.0e-9          # Scalar amplitude [dimensionless]
    ns = 0.965           # Scalar spectral index [dimensionless]
    kmax_calc = 2.0      # Maximum k for CAMB internal calculation [Mpc^-1]

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

    # Ensure linear matter power spectrum
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

    # Extract P(k) at z=0
    pk_at_z0 = pk_out[0, :]

    # Save to CSV
    output_path = os.path.join(output_dir, "result.csv")
    output_df = pd.DataFrame({'kh': kh, 'P_k': pk_at_z0})
    output_df.to_csv(output_path, index=False, float_format='%.6e')

    # Print summary to console
    print("Linear matter power spectrum P(k) at z=0 saved to " + output_path)
    print("kh (h/Mpc) range: " + ("%.6e" % kh.min()) + " to " + ("%.6e" % kh.max()))
    print("P_k ((Mpc/h)^3) range: " + ("%.6e" % pk_at_z0.min()) + " to " + ("%.6e" % pk_at_z0.max()))
    print("Number of k points: " + str(len(kh)))
    print("\nFirst 5 rows of the output:")
    print(output_df.head())


if __name__ == "__main__":
    compute_linear_matter_power_spectrum()