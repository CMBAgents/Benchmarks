# filename: codebase/matter_power_spectrum.py
import os
import camb
import numpy as np
import pandas as pd

def calculate_matter_power_spectrum():
    r"""
    Calculates the linear matter power spectrum P(k) at redshift z=0
    for a flat Lambda CDM cosmology using specified parameters with CAMB.

    The results (kh and P(k)) are saved to a CSV file.
    kh: Wavenumber (in h/Mpc)
    P_k: Linear matter power spectrum (in (Mpc/h)^3)
    """

    # Cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Baryon density Omega_b * h^2
    omch2 = 0.122  # Cold dark matter density Omega_c * h^2
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature Omega_k
    tau = 0.06  # Optical depth to reionization
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    k_max_camb_internal = 2.0  # k_max for CAMB internal calculations in Mpc^-1

    print("Setting CAMB parameters...")
    print("H0: " + str(H0) + " km/s/Mpc")
    print("ombh2: " + str(ombh2))
    print("omch2: " + str(omch2))
    print("mnu: " + str(mnu) + " eV")
    print("omk: " + str(omk))
    print("tau: " + str(tau))
    print("As: " + str(As))
    print("ns: " + str(ns))
    print("k_max (for CAMB internal): " + str(k_max_camb_internal) + " Mpc^-1")
    
    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau, num_massive_neutrinos=3)
    pars.InitPower.As = As
    pars.InitPower.ns = ns
    
    # We want the matter power spectrum, not CMB C_ls
    pars.WantCls = False
    pars.WantScalars = True
    pars.WantTensors = False
    pars.WantVectors = False

    # Calculate linear power spectrum
    pars.NonLinear = camb.model.NonLinear_none 
    
    # Set redshifts and k_max for CAMB's internal calculation
    # kmax here is in Mpc^-1 units
    pars.set_matter_power(redshifts=[0.0], kmax=k_max_camb_internal)

    print("\nRunning CAMB to get results...")
    results = camb.get_results(pars)

    # Define k range for output power spectrum
    # kh_min, kh_max are in h/Mpc units
    kh_output_min = 1e-4  # h/Mpc
    kh_output_max = 1.0   # h/Mpc
    n_kh_points = 200

    print("Calculating linear matter power spectrum for z=0.")
    print("Output kh range: " + str(kh_output_min) + " h/Mpc to " + str(kh_output_max) + " h/Mpc, " + str(n_kh_points) + " points.")

    # Get matter power spectrum P(k)
    # hubble_units=True means P(k) is in (Mpc/h)^3
    # k_hunit=True means k is in h/Mpc
    # nonlinear=False explicitly requests linear P(k)
    kh_values, z_values, pk_values = results.get_matter_power_spectrum(
        minkh=kh_output_min, 
        maxkh=kh_output_max, 
        npoints=n_kh_points,
        var1='delta_tot', var2='delta_tot',  # Total matter density perturbations
        hubble_units=True,  # P(k) in (Mpc/h)^3
        k_hunit=True,       # k in h/Mpc
        nonlinear=False     # Ensure linear power spectrum
    )
    
    # pk_values is a 2D array (n_redshifts, n_k_points). We need P(k) at z=0.
    # Since we requested only z=0, z_values will be [0.] and pk_values[0] is P(k) at z=0.
    P_k_at_z0 = pk_values[0]

    # Create a Pandas DataFrame
    df = pd.DataFrame({'kh': kh_values, 'P_k': P_k_at_z0})

    # Save to CSV
    output_dir = 'data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    csv_filename = os.path.join(output_dir, 'result.csv')
    df.to_csv(csv_filename, index=False, float_format='%.8e')

    print("\nResults saved to: " + str(csv_filename))
    
    # Print summary of the results
    print("\nFirst 5 rows of the result:")
    print(df.head().to_string())
    print("\nLast 5 rows of the result:")
    print(df.tail().to_string())

    # Print some specific values for verification
    if len(df) > 0:
        print("\nExample values:")
        print("P(k) at min kh (" + str(df['kh'].iloc[0]) + " h/Mpc): " + str(df['P_k'].iloc[0]) + " (Mpc/h)^3")
        if len(df) > 1:
             print("P(k) at max kh (" + str(df['kh'].iloc[-1]) + " h/Mpc): " + str(df['P_k'].iloc[-1]) + " (Mpc/h)^3")
        if len(df) >= n_kh_points // 2:
            mid_index = n_kh_points // 2
            print("P(k) at mid kh (" + str(df['kh'].iloc[mid_index]) + " h/Mpc): " + str(df['P_k'].iloc[mid_index]) + " (Mpc/h)^3")

if __name__ == '__main__':
    try:
        calculate_matter_power_spectrum()
    except ImportError as e:
        print("ImportError: " + str(e))
        print("Please ensure CAMB is installed. You might need to run: pip install camb")
    except Exception as e:
        print("An error occurred:")
        import traceback
        traceback.print_exc()
