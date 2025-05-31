# filename: codebase/relative_matter_power_spectrum.py
import os
import camb
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def ensure_data_dir(directory="data"):
    """
    Ensures that the specified directory exists, creating it if necessary.

    Args:
        directory (str): The path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    print("Directory " + directory + " ensured.")


def get_pk_interpolated(hierarchy_type, H0_val, ombh2_val, omch2_val, sum_mnu_val, As_val, ns_val, k_target_array):
    """
    Computes the linear matter power spectrum for a given neutrino hierarchy and
    interpolates it onto a target array of k values.

    Args:
        hierarchy_type (str): Neutrino hierarchy ('normal' or 'inverted').
        H0_val (float): Hubble constant (km/s/Mpc).
        ombh2_val (float): Baryon density parameter.
        omch2_val (float): Cold dark matter density parameter.
        sum_mnu_val (float): Sum of neutrino masses (eV).
        As_val (float): Scalar amplitude.
        ns_val (float): Scalar spectral index.
        k_target_array (np.ndarray): Array of k values (in h/Mpc) onto which P(k) should be interpolated.

    Returns:
        tuple: (np.ndarray, np.ndarray)
            - k_target_array (h/Mpc)
            - Interpolated P(k) values ((Mpc/h)^3)
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0_val, ombh2=ombh2_val, omch2=omch2_val, mnu=sum_mnu_val, omk=0)
    pars.InitPower.set_params(As=As_val, ns=ns_val)
    
    pars.num_massive_neutrinos = 3
    pars.neutrino_hierarchy = hierarchy_type
    
    pars.WantCls = False  # We don't need CMB C_ls
    pars.WantScalars = True
    pars.WantTensors = False
    pars.DoLensing = False

    # Define k-range for CAMB's internal calculation (denser and wider for good interpolation)
    # k_target_array is already in h/Mpc
    minkh_calc = max(1e-5, k_target_array[0] * 0.9)  # h/Mpc
    maxkh_calc = k_target_array[-1] * 1.1  # h/Mpc
    npoints_calc = 1000  # Number of log-spaced points for CAMB's internal P(k)

    pars.Transfer.high_precision = True
    # transfer_kmax is in h/Mpc
    pars.Transfer.kmax = maxkh_calc + 1.0  # Ensure transfer functions are computed up to a high enough k
    
    # Set matter power spectrum parameters for CAMB calculation
    # kmax for set_matter_power is in h/Mpc if hubble_units=True (default)
    pars.set_matter_power(redshifts=[0.0], kmax=maxkh_calc)
    
    results = camb.get_results(pars)
    
    # Get P(k) from CAMB on its internal log-spaced grid
    # kh_camb is in h/Mpc, pk_camb_raw is in (Mpc/h)^3
    kh_camb, z_array, pk_camb_raw = results.get_matter_power_spectrum(
        minkh=minkh_calc, maxkh=maxkh_calc, npoints=npoints_calc,
        var1='delta_tot', var2='delta_tot',
        hubble_units=True, k_hunit=True,
        nonlinear=False, redshifts=[0.0]
    )
    
    # Interpolate P(k) onto the target linearly spaced k-values
    # pk_camb_raw[0] corresponds to z=0
    pk_interpolated = np.interp(k_target_array, kh_camb, pk_camb_raw[0])
    
    return k_target_array, pk_interpolated


def main():
    """
    Main function to calculate, save, and plot the relative difference
    in matter power spectra.
    """
    ensure_data_dir("data")
    
    # Cosmological parameters
    H0 = 67.5  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Baryon density
    omch2 = 0.122  # Cold dark matter density
    sum_mnu = 0.11  # Neutrino mass sum in eV
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    # omk = 0 is set implicitly in set_cosmology by not providing omk or by mnu, ombh2, omch2

    h_param = H0 / 100.0

    # k-range for power spectrum as per problem statement
    # Range is for k_phys * h: 10^-4 < k_phys * h < 2 (Mpc^-1)
    # CAMB uses k/h (call it k_camb). So k_phys = k_camb * h.
    # The range becomes: 10^-4 < (k_camb * h) * h < 2  => 10^-4 < k_camb * h^2 < 2
    # So, k_camb_min = 10^-4 / h^2 and k_camb_max = 2 / h^2
    
    kh_min_problem = 1e-4  # Mpc^-1
    kh_max_problem = 2.0   # Mpc^-1
    npoints_target = 200

    k_div_h_min_target = kh_min_problem / (h_param**2)  # units: h/Mpc
    k_div_h_max_target = kh_max_problem / (h_param**2)  # units: h/Mpc
    
    # Linearly spaced k values for the output CSV and plot (in h/Mpc)
    k_target_values = np.linspace(k_div_h_min_target, k_div_h_max_target, npoints_target)  # h/Mpc

    print("Calculating P(k) for normal hierarchy...")
    _, pk_normal = get_pk_interpolated('normal', H0, ombh2, omch2, sum_mnu, As, ns, k_target_values)
    
    print("Calculating P(k) for inverted hierarchy...")
    _, pk_inverted = get_pk_interpolated('inverted', H0, ombh2, omch2, sum_mnu, As, ns, k_target_values)
    
    # Calculate relative difference
    # Ensure pk_normal is not zero to avoid division by zero, though highly unlikely
    if np.any(pk_normal == 0):
        print("Warning: pk_normal contains zero values. Relative difference might be ill-defined.")
        # Handle as appropriate, e.g., by replacing zeros or exiting
        rel_diff = np.full_like(pk_normal, np.nan)  # Or some other error indicator
        valid_mask = pk_normal != 0
        rel_diff[valid_mask] = pk_inverted[valid_mask] / pk_normal[valid_mask] - 1
    else:
        rel_diff = pk_inverted / pk_normal - 1
        
    # Save results to CSV
    csv_filename = "data/result.csv"
    output_data = np.vstack((k_target_values, rel_diff)).T
    header_str = "k (h/Mpc),rel_diff ((P_inv/P_nor)-1)"
    np.savetxt(csv_filename, output_data, header=header_str, delimiter=',', comments='')
    print("Results saved to: " + csv_filename)
    
    # Print some numerical results
    print("\n--- Numerical Results ---")
    print("k range (h/Mpc): " + str(np.min(k_target_values)) + " to " + str(np.max(k_target_values)))
    print("Relative difference range: " + str(np.min(rel_diff)) + " to " + str(np.max(rel_diff)))
    
    # Plotting
    plt.rcParams['text.usetex'] = False  # Avoid LaTeX rendering issues
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogx(k_target_values, rel_diff)  # Use semilogx for k-axis
    
    ax.set_xlabel("k (h/Mpc)")
    ax.set_ylabel("Relative Difference ((P_inv / P_nor) - 1)")
    ax.set_title("Relative Difference in P(k) (Inverted vs Normal Hierarchy)")
    ax.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    plot_filename = "data/relative_matter_power_spectrum_difference_1_" + timestamp + ".png"
    plt.savefig(plot_filename, dpi=300)
    print("\nPlot saved to: " + plot_filename)
    print("Plot description: Shows the relative difference in the linear matter power spectrum " + \
          "((P_inverted / P_normal) - 1) at z=0 as a function of wavenumber k (in h/Mpc). " + \
          "The x-axis (k) is on a logarithmic scale.")


if __name__ == '__main__':
    main()
