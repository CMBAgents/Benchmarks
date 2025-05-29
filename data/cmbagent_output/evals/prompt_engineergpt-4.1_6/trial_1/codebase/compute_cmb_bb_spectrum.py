# filename: codebase/compute_cmb_bb_spectrum.py
import os
import numpy as np
import pandas as pd

# Ensure data directory exists
database_path = "data"
if not os.path.exists(database_path):
    os.makedirs(database_path)

try:
    import camb
    from camb import model, initialpower
except ImportError as e:
    print("CAMB is not installed. Please install camb and rerun the code.")
    raise e


def compute_cmb_bb_spectrum(
    H0=67.5,                # Hubble constant [km/s/Mpc]
    ombh2=0.022,            # Baryon density [dimensionless]
    omch2=0.122,            # Cold dark matter density [dimensionless]
    mnu=0.06,               # Neutrino mass sum [eV]
    omk=0.0,                # Curvature [dimensionless]
    tau=0.06,               # Optical depth to reionization [dimensionless]
    r=0.1,                  # Tensor-to-scalar ratio [dimensionless]
    As=2e-9,                # Scalar amplitude [dimensionless]
    ns=0.965,               # Scalar spectral index [dimensionless]
    lmin=2,                 # Minimum multipole
    lmax=3000               # Maximum multipole
):
    r"""
    Compute the CMB B-mode polarization power spectrum (C_l^{BB}) using CAMB.

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
    r : float
        Tensor-to-scalar ratio.
    As : float
        Scalar amplitude.
    ns : float
        Scalar spectral index.
    lmin : int
        Minimum multipole moment.
    lmax : int
        Maximum multipole moment.

    Returns
    -------
    l : ndarray
        Array of multipole moments (l).
    BB : ndarray
        B-mode polarization power spectrum (C_l^{BB}) in microKelvin^2.
    """
    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=r)
    pars.WantTensors = True
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.Want_CMB = True
    pars.Want_CMB_lensing = False

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
    # 'total' includes tensor and scalar, 'tensor' is tensor-only
    # For raw Cl, the shape is (lmax+1, 4): columns are TT, EE, BB, TE

    # Extract BB spectrum
    cl = powers['total']  # shape: (lmax+1, 4)
    # cl[:,2] is BB
    l_arr = np.arange(cl.shape[0])  # l = 0 ... lmax
    BB = cl[:,2]  # BB in muK^2

    # Restrict to lmin <= l <= lmax
    mask = (l_arr >= lmin) & (l_arr <= lmax)
    l_out = l_arr[mask]
    BB_out = BB[mask]

    return l_out, BB_out


if __name__ == "__main__":
    # Compute the spectrum
    l, BB = compute_cmb_bb_spectrum()

    # Save to CSV
    df = pd.DataFrame({'l': l, 'BB': BB})
    output_csv = os.path.join(database_path, "result.csv")
    df.to_csv(output_csv, index=False)

    # Print summary of results
    np.set_printoptions(precision=6, suppress=True)
    print("CMB B-mode polarization power spectrum (C_l^{BB}) computed for l=2 to l=3000.")
    print("Units: BB in microKelvin^2.")
    print("Results saved to " + output_csv)
    print("\nFirst 5 rows:")
    print(df.head(5).to_string(index=False))
    print("\nLast 5 rows:")
    print(df.tail(5).to_string(index=False))