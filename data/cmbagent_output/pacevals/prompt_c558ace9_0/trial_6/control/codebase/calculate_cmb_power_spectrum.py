# filename: codebase/calculate_cmb_power_spectrum.py
import camb
import numpy as np
import traceback  # For detailed error reporting

# Ensure matplotlib does not use LaTeX, as per general instructions.
# This is set globally even if this specific script doesn't plot.
import matplotlib
matplotlib.rcParams['text.usetex'] = False


def calculate_cmb_power_spectrum():
    """
    Calculates the CMB raw temperature power spectrum (ClTT) for a flat Lambda CDM cosmology
    using specified parameters with CAMB.

    The function sets up cosmological parameters, runs CAMB, and extracts the unlensed
    scalar temperature power spectrum ClTT in units of muK^2 for multipole moments
    from l=2 to l=3000.

    Cosmological Parameters:
        Hubble constant (H0): 74 km/s/Mpc
        Baryon density (ombh2): 0.022 (dimensionless Omega_b * h^2)
        Cold dark matter density (omch2): 0.122 (dimensionless Omega_c * h^2)
        Neutrino mass sum (mnu): 0.06 eV
        Curvature (omk): 0 (dimensionless Omega_k)
        Optical depth to reionization (tau): 0.06 (dimensionless)
        Scalar amplitude (As): 2.0e-9 (dimensionless)
        Scalar spectral index (ns): 0.965 (dimensionless)
        Maximum multipole (lmax): 3000

    Returns:
        tuple: (ell, TT_spectrum)
            ell (numpy.ndarray): Array of multipole moments (l) from 2 to 3000.
            TT_spectrum (numpy.ndarray): Array of raw temperature power spectrum (ClTT) values in muK^2.
            Returns (None, None) if an error occurs.
    """
    # Define cosmological parameters
    H0 = 74.0  # Hubble constant in km/s/Mpc
    ombh2 = 0.022  # Physical baryon density (Omega_b * h^2)
    omch2 = 0.122  # Physical cold dark matter density (Omega_c * h^2)
    mnu = 0.06  # Sum of neutrino masses in eV
    omk = 0.0  # Curvature density (Omega_k)
    tau = 0.06  # Optical depth to reionization
    As = 2.0e-9  # Scalar amplitude
    ns = 0.965  # Scalar spectral index
    lmax = 3000  # Maximum multipole moment for calculation

    # CAMB settings
    pars = camb.CAMBparams()

    try:
        # Set cosmological parameters
        # num_massive_neutrinos=3 assumes the sum mnu is distributed among 3 massive neutrino species.
        # Neff=3.046 is the standard effective number of relativistic species.
        pars.set_cosmology(
            H0=H0, 
            ombh2=ombh2, 
            omch2=omch2, 
            mnu=mnu, 
            omk=omk, 
            tau=tau, 
            num_massive_neutrinos=3, 
            Neff=3.046
        )

        # Set initial power spectrum parameters
        pars.InitPower.set_params(As=As, ns=ns)

        # Set calculation options for lmax
        # lens_potential_accuracy=0 for unlensed scalar spectra (raw spectra)
        # max_eta_k_scalar is increased for higher l accuracy, as per CAMB examples for lmax > 2000
        pars.set_for_lmax(lmax=lmax, lens_potential_accuracy=0, max_eta_k_scalar=float(lmax * 2.5)) 
        
        # Specify that we want scalar spectra and do not need tensor spectra for this task.
        # WantScalars is True by default, but explicit setting is fine.
        pars.WantScalars = True
        pars.WantTensors = False 
        
        # Run CAMB calculation
        print("Running CAMB to calculate power spectra...")
        results = camb.get_results(pars)
        print("CAMB calculation finished.")

        # Get power spectra
        # CMB_unit='muK' ensures output Cls are in muK^2.
        # raw_cl=True ensures Cl rather than l(l+1)C_l/2pi.
        # lmax here specifies the maximum l to retrieve from the results.
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', lmax=lmax, raw_cl=True)
        
        # Extract unlensed scalar Cls
        # 'unlensed_scalar' contains the unlensed scalar Cls.
        # This is a numpy array with columns: l, TT, EE, BB, TE (if computed).
        # The columns are L, ClTT, ClEE, ClBB, ClTE.
        cls_unlensed_scalar = powers['unlensed_scalar']
        
        # The array typically starts from l=0. We need l from 2 to lmax.
        # So, we slice from index 2 up to index lmax (inclusive for l).
        # l values are in the first column (index 0).
        # TT values are in the second column (index 1) (ClTT in muK^2).
        
        if cls_unlensed_scalar.shape[0] < lmax + 1:
            raise ValueError("CAMB returned fewer multipoles than requested (lmax=" + str(lmax) + "). Array shape: " + str(cls_unlensed_scalar.shape))

        # Extract relevant part of the array: l from 2 to lmax
        # Index 0 corresponds to l=0, index 1 to l=1, index 2 to l=2.
        # Slice [2:lmax+1] includes indices from 2 up to lmax.
        ell = cls_unlensed_scalar[2:lmax+1, 0].astype(int)  # Multipole moment l
        TT_spectrum = cls_unlensed_scalar[2:lmax+1, 1]    # TT power spectrum (ClTT) in muK^2

        print("Successfully calculated CMB TT power spectrum.")
        # Shape should be (2999,) for l=2...3000
        print("Shape of l array (multipoles): " + str(ell.shape)) 
        print("Shape of TT spectrum array: " + str(TT_spectrum.shape))
        
        print("\nSample of calculated CMB TT power spectrum values (in muK^2):")
        print("First 5 values:")
        for i in range(min(5, len(ell))):
            print("l = " + str(ell[i]) + ", TT = " + str(TT_spectrum[i]))

        print("\nLast 5 values:")
        start_index_for_last_values = max(0, len(ell) - 5)
        for i in range(start_index_for_last_values, len(ell)):
            print("l = " + str(ell[i]) + ", TT = " + str(TT_spectrum[i]))
            
        return ell, TT_spectrum

    except Exception:
        print("An error occurred during CAMB calculation or data processing:")
        print(traceback.format_exc())  # Print full traceback for debugging
        return None, None


if __name__ == '__main__':
    print("Starting CMB power spectrum calculation using CAMB...")
    ell_values, tt_values = calculate_cmb_power_spectrum()

    if ell_values is not None and tt_values is not None:
        print("\nCMB power spectrum calculation successful and data is ready.")
        if len(ell_values) > 0:
            print("Multipole moment l ranges from " + str(ell_values[0]) + " to " + str(ell_values[-1]) + ".")
        else:
            print("No multipole moments were generated (ell_values array is empty).")
        print("TT spectrum values (in muK^2) are available for further processing.")
    else:
        print("\nCMB power spectrum calculation failed. Please check error messages above.")
