# filename: codebase/camb_bb_spectrum.py
import os
import numpy as np
import pandas as pd

# Ensure data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

try:
    import camb
    from camb import model, initialpower
except ImportError as e:
    print("CAMB package is required but not installed. Please install camb and rerun the code.")
    raise e


def compute_cmb_bb_spectrum():
    r"""Compute the CMB B-mode polarization power spectrum (C_l^{BB}) for a flat Lambda CDM cosmology using CAMB, with the following parameters:
        - Hubble constant (H0): 67.5 km/s/Mpc
        - Baryon density (ombh2): 0.022
        - Cold dark matter density (omch2): 0.122
        - Neutrino mass sum (mnu): 0.06 eV
        - Curvature (omk): 0
        - Optical depth to reionization (tau): 0.06
        - Tensor-to-scalar ratio (r): 0.1
        - Scalar amplitude (As): 2e-9
        - Scalar spectral index (ns): 0.965

    Returns:
        None. Saves the result as 'data/result.csv' with columns:
            - l: Multipole moment (integer, 2 to 3000)
            - BB: B-mode polarization power spectrum (C_l^{BB} in microK^2)
    """
    # Cosmological parameters
    H0 = 67.5  # Hubble constant [km/s/Mpc]
    ombh2 = 0.022  # Baryon density
    omch2 = 0.122  # Cold dark matter density
    mnu = 0.06  # Neutrino mass sum [eV]
    omk = 0.0  # Curvature
    tau = 0.06  # Optical depth to reionization
    r_tensor = 0.1  # Tensor-to-scalar ratio
    As = 2e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index

    # Multipole range
    lmin = 2
    lmax = 3000

    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=r_tensor)
    pars.WantTensors = True
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    pars.Want_CMB = True

    # Calculate results
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
    # 'raw_cl=True' returns Cl in K^2, so we need to convert to microK^2

    # Extract the total power spectrum (includes tensor modes)
    cl = powers['total']  # shape: (lmax+1, 4), columns: TT, EE, BB, TE

    # l values: cl[ell, ...] for ell = 0, 1, ..., lmax
    # We want l = 2 to lmax
    ells = np.arange(cl.shape[0])
    mask = (ells >= lmin) & (ells <= lmax)
    l_vals = ells[mask]
    # BB is column 2
    cl_bb = cl[mask, 2]  # [K^2]

    # Convert from K^2 to microK^2: (1 K = 1e6 microK) => (K^2 = 1e12 microK^2)
    cl_bb_muK2 = cl_bb * 1e12

    # Save to CSV
    df = pd.DataFrame({'l': l_vals, 'BB': cl_bb_muK2})
    output_path = os.path.join("data", "result.csv")
    df.to_csv(output_path, index=False)

    # Print summary
    print("CMB B-mode polarization power spectrum (C_l^{BB}) computed for l = 2 to 3000.")
    print("Results saved to data/result.csv")
    print("First 10 rows:")
    print(df.head(10).to_string(index=False))
    print("\nUnits: l (dimensionless), BB (microK^2)")


if __name__ == "__main__":
    compute_cmb_bb_spectrum()