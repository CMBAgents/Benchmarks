# filename: codebase/cmb_power_spectrum.py
import camb
import numpy as np
import pandas as pd
import os

def calculate_cmb_power_spectrum():
    """
    Calculates the CMB raw temperature power spectrum for a flat Lambda CDM cosmology
    using specified parameters with CAMB and saves the results to a CSV file.
    """
    try:
        # Cosmological parameters
        H0_val = 74.0       # Hubble constant in km/s/Mpc
        ombh2_val = 0.022   # Physical baryon density Omega_b h^2
        omch2_val = 0.122   # Physical cold dark matter density Omega_c h^2
        mnu_val = 0.06      # Sum of neutrino masses in eV
        omk_val = 0.0       # Curvature Omega_k (0 for flat universe)
        tau_val = 0.06      # Optical depth to reionization
        As_val = 2.0e-9     # Scalar amplitude (dimensionless)
        ns_val = 0.965      # Scalar spectral index (dimensionless)

        lmax_calc = 3000    # Maximum multipole moment l for calculation

        # 1. Set up CAMB parameters
        print("Setting CAMB parameters...")
        pars = camb.CAMBparams()
        
        # Set cosmological parameters
        # H0: Hubble constant in km/s/Mpc. Cosmological significance: Current expansion rate of the Universe.
        # ombh2: Physical baryon density. Cosmological significance: Fraction of energy density in baryonic matter.
        # omch2: Physical cold dark matter density. Cosmological significance: Fraction of energy density in cold dark matter.
        # mnu: Sum of neutrino masses in eV. Cosmological significance: Affects structure formation and CMB.
        # omk: Curvature density. Cosmological significance: Determines the geometry of the Universe (0 for flat).
        # tau: Optical depth to reionization. Cosmological significance: Scattering of CMB photons by free electrons during reionization.
        pars.set_cosmology(H0=H0_val, ombh2=ombh2_val, omch2=omch2_val, mnu=mnu_val, omk=omk_val, tau=tau_val)
        
        # Set initial power spectrum parameters
        # As: Scalar amplitude. Cosmological significance: Amplitude of primordial scalar density perturbations.
        # ns: Scalar spectral index. Cosmological significance: Describes the scale dependence of primordial scalar perturbations.
        pars.InitPower.set_params(As=As_val, ns=ns_val)
        
        # Set lmax for calculation and specify unlensed spectra
        # lmax: Maximum multipole moment.
        # lens_potential_accuracy=0: Calculate unlensed (raw) CMB power spectra.
        pars.set_for_lmax(lmax=lmax_calc, lens_potential_accuracy=0)
        print("CAMB parameters set.")

        # 2. Run CAMB to get results
        print("Running CAMB calculations...")
        results = camb.get_results(pars)
        print("CAMB calculations finished.")

        # 3. Get CMB power spectra (raw Cl in muK^2)
        # 'unlensed_scalar' contains TT, EE, BB, TE spectra. We need TT (index 0).
        # CMB_unit='muK': Output spectra in microKelvin^2.
        # raw_cl=True: Output C_l directly, not l(l+1)C_l/(2pi).
        print("Extracting CMB power spectra...")
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
        # cl_TT_all is an array of C_l^TT values from l=0 to l=lmax_calc
        # Units: muK^2
        cl_TT_all = powers['unlensed_scalar'][0, :]
        print("CMB power spectra extracted.")

        # 4. Prepare data for CSV output (l=2 to lmax_calc)
        # Multipole moments (dimensionless)
        ls_for_csv = np.arange(2, lmax_calc + 1)
        # Corresponding C_l^TT values (muK^2)
        cl_TT_for_csv = cl_TT_all[2:lmax_calc + 1]

        # 5. Save to CSV file
        output_dir = "data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        file_path = os.path.join(output_dir, "result.csv")
        
        df_results = pd.DataFrame({'l': ls_for_csv, 'TT': cl_TT_for_csv})
        df_results.to_csv(file_path, index=False, header=['l', 'TT'])
        print("Results saved to " + file_path)
        
        # Print some sample data
        print("\nSample of the first 5 rows of the C_l^TT data:")
        print(df_results.head())
        print("\nSample of the last 5 rows of the C_l^TT data:")
        print(df_results.tail())
        
        print("\nSummary of C_l^TT data:")
        print("Number of l values: " + str(len(df_results)))
        print("Min l: " + str(df_results['l'].min()))
        print("Max l: " + str(df_results['l'].max()))
        print("Min TT (muK^2): " + str(df_results['TT'].min()))
        print("Max TT (muK^2): " + str(df_results['TT'].max()))
        print("Mean TT (muK^2): " + str(df_results['TT'].mean()))


    except Exception as e:
        print("An error occurred:")
        print(str(e))


if __name__ == '__main__':
    calculate_cmb_power_spectrum()
