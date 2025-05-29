# filename: codebase/linear_matter_power_spectrum.py
import os
import numpy as np
import camb
from camb import model

def compute_linear_matter_power_spectrum():
    """
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
    H0 = 67.5           # Hubble constant [km/s/Mpc]
    ombh2 = 0.022       # Omega_b * h^2 [dimensionless]
    omch2 = 0.122       # Omega_c * h^2 [dimensionless]
    mnu = 0.06          # Sum of neutrino masses [eV]
    omk = 0.0           # Curvature [dimensionless]
    tau = 0.06          # Optical depth to reionization [dimensionless]
    As = 2.0e-9         # Scalar amplitude [dimensionless]
    ns = 0.965          # Scalar spectral index [dimensionless]
    kmax_calc = 2.0     # Maximum k for CAMB calculation [Mpc^-1]

    # Output P(k) specifications
    redshift_output = 0.0
    minkh_output = 1e-4     # Minimum k/h for output [h/Mpc]
    maxkh_output = 1.0      # Maximum k/h for output [h/Mpc]
    npoints_output = 200    # Number of k points

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
        WantTransfer=True
    )

    # Configure matter power spectrum calculation
    pars.set_matter_power(redshifts=[redshift_output], kmax=kmax_calc)
    pars.NonLinear = model.NonLinear_none

    # Run CAMB calculation
    results = camb.get_results(pars)

    # Extract the linear matter power spectrum
    kh, z_out, pk = results.get_matter_power_spectrum(
        minkh=minkh_output,
        maxkh=maxkh_output,
        npoints=npoints_output
    )
    # pk shape: (len(z_out), len(kh)), here z_out = [0.0]
    P_k_values = pk[0, :]

    # Save results to CSV
    output_data = np.column_stack((kh, P_k_values))
    header_csv = "kh,P_k"
    output_path = os.path.join(output_dir, "result.csv")
    np.savetxt(output_path, output_data, delimiter=",", header=header_csv, comments="")

    # Print summary to console
    np.set_printoptions(precision=6, suppress=True)
    print("Linear matter power spectrum at z=0 calculated and saved to " + output_path)
    print("kh range: " + str(kh[0]) + " to " + str(kh[-1]) + " [h/Mpc]")
    print("P(k) range: " + str(P_k_values.min()) + " to " + str(P_k_values.max()) + " [(Mpc/h)^3]")
    print("First 5 rows (kh [h/Mpc], P_k [(Mpc/h)^3]):")
    for i in range(5):
        print(str(kh[i]) + ", " + str(P_k_values[i]))


if __name__ == "__main__":
    compute_linear_matter_power_spectrum()