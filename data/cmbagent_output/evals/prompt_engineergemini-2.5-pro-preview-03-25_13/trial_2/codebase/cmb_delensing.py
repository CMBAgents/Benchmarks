# filename: codebase/cmb_delensing.py
import os
import numpy as np
import pandas as pd
try:
    import camb
    from camb import model  # model is used by camb.CAMBparams, not directly here
except ImportError:
    print("CAMB is not installed. Please install it to run this script.")
    # This script will fail later if camb is not available.
    # Another agent is expected to handle installation.
    pass


class CMBDelensingCalculator:
    r"""
    Calculates the delensing efficiency of CMB B-mode polarization.

    The calculation involves:
    1. Computing lensed CMB power spectra using CAMB.
    2. Computing the CMB lensing potential power spectrum.
    3. Loading a lensing noise power spectrum.
    4. Calculating the residual lensing potential power spectrum after Wiener filtering.
    5. Computing the delensed CMB B-mode power spectrum using the residual potential.
    6. Determining the delensing efficiency.
    """

    def __init__(self, H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06,
                 As=2e-9, ns=0.965, noise_file_path=None):
        r"""
        Initializes the calculator with cosmological parameters and noise file path.

        Parameters
        ----------
        H0 : float
            Hubble constant in km/s/Mpc.
        ombh2 : float
            Baryon density parameter.
        omch2 : float
            Cold dark matter density parameter.
        mnu : float
            Sum of neutrino masses in eV.
        omk : float
            Curvature density parameter.
        tau : float
            Optical depth to reionization.
        As : float
            Scalar amplitude.
        ns : float
            Scalar spectral index.
        noise_file_path : str
            Path to the CSV file containing lensing noise power spectrum (l, Nl).
            Nl should be N_0 * (l(l+1))^2 / (2pi).
        """
        self.H0 = H0
        self.ombh2 = ombh2
        self.omch2 = omch2
        self.mnu = mnu
        self.omk = omk
        self.tau = tau
        self.As = As
        self.ns = ns
        self.noise_file_path = noise_file_path

        self.lmax_calc = 2000  # Max ell for C_l^phiphi and N_l^phiphi processing
        self.lmax_spectra = 2500 # Max ell for CAMB C_l^BB computations

        self.ells = np.arange(self.lmax_calc + 1)
        self.phi_norm_factor = (self.ells * (self.ells + 1))**2 / (2 * np.pi)
        # For ells=0, ells*(ells+1) is 0. phi_norm_factor[0] will be 0.
        # For ells=1, ells*(ells+1) is 2. phi_norm_factor[1] will be (2^2)/(2pi) = 2/pi.
        # C_phi[0] and C_phi[1] are zero, so scaled versions will also be zero.

        self.inv_phi_norm_factor = np.zeros_like(self.phi_norm_factor, dtype=float)
        valid_ells_mask = (self.ells >= 2)
        # Avoid division by zero for ells < 2
        self.inv_phi_norm_factor[valid_ells_mask] = (2 * np.pi) / (self.ells[valid_ells_mask] * (self.ells[valid_ells_mask] + 1))**2
        
        os.makedirs("data", exist_ok=True)

    def _get_camb_pars(self):
        r"""Configures and returns CAMBparams object."""
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=self.H0, ombh2=self.ombh2, omch2=self.omch2,
                           mnu=self.mnu, omk=self.omk, tau=self.tau)
        pars.InitPower.set_params(As=self.As, ns=self.ns)
        pars.WantTensors = False 
        pars.DoLensing = True
        pars.set_for_lmax(lmax=self.lmax_spectra, lens_potential_accuracy=1)
        return pars

    def calculate_lensed_cl_bb(self, pars, camb_results):
        r"""
        Calculates the lensed B-mode power spectrum C_ell^BB.

        Parameters
        ----------
        pars : camb.CAMBparams
            Configured CAMB parameters (used for units and raw_cl interpretation).
        camb_results : camb.results.CAMBdata
            Results object from a CAMB run.

        Returns
        -------
        numpy.ndarray
            Lensed C_ell^BB in muK^2, up to self.lmax_spectra.
        """
        powers_raw = camb_results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
        # BB is index 2: TT, EE, BB, TE
        cl_bb_lensed = powers_raw['lensed_scalar'][:, 2] 
        return cl_bb_lensed

    def calculate_cl_phiphi_scaled_from_results(self, camb_results):
        r"""
        Calculates C_ell^phiphi * (l(l+1))^2 / (2pi) from a CAMB results object.

        Parameters
        ----------
        camb_results : camb.results.CAMBdata
            Results object from a CAMB run.

        Returns
        -------
        numpy.ndarray
            Scaled C_ell^phiphi, up to self.lmax_calc.
        """
        # get_lens_potential_cls returns raw C_ell^phiphi
        cl_phiphi_raw = camb_results.get_lens_potential_cls(lmax=self.lmax_calc)[:, 0] 
        
        cl_phiphi_scaled = np.zeros(self.lmax_calc + 1)
        len_to_use = min(len(cl_phiphi_raw), self.lmax_calc + 1)
        cl_phiphi_scaled[:len_to_use] = cl_phiphi_raw[:len_to_use] * self.phi_norm_factor[:len_to_use]
        
        return cl_phiphi_scaled

    def load_noise_nl_phiphi(self):
        r"""
        Loads the scaled noise power spectrum N_ell^phiphi = N_0 * (l(l+1))^2 / (2pi).

        Returns
        -------
        numpy.ndarray
            N_ell^phiphi (scaled noise), up to self.lmax_calc.
        """
        if self.noise_file_path is None:
            raise ValueError("Noise file path is not provided.")
        if not os.path.exists(self.noise_file_path):
            raise FileNotFoundError("Noise file not found at: " + str(self.noise_file_path))
            
        noise_data = pd.read_csv(self.noise_file_path)
        l_noise = noise_data['l'].values.astype(int)
        nl_phiphi_scaled_values = noise_data['Nl'].values

        nl_phiphi_scaled_arr = np.zeros(self.lmax_calc + 1)
        
        valid_mask = (l_noise <= self.lmax_calc) & (l_noise >= 0)
        nl_phiphi_scaled_arr[l_noise[valid_mask]] = nl_phiphi_scaled_values[valid_mask]
        
        return nl_phiphi_scaled_arr

    def calculate_residual_cl_phiphi(self, cl_phiphi_scaled, nl_phiphi_scaled):
        r"""
        Calculates the residual C_ell^phiphi (unscaled) for CAMB input.

        Parameters
        ----------
        cl_phiphi_scaled : numpy.ndarray
            Scaled C_ell^phiphi.
        nl_phiphi_scaled : numpy.ndarray
            Scaled N_ell^phiphi (noise).

        Returns
        -------
        numpy.ndarray
            Residual C_ell^phiphi (unscaled), padded to self.lmax_spectra.
        """
        cl_pp_s = cl_phiphi_scaled[:self.lmax_calc+1]
        n_ell_pp_s = nl_phiphi_scaled[:self.lmax_calc+1]

        cl_pp_res_scaled = np.zeros(self.lmax_calc + 1)
        denominator = cl_pp_s + n_ell_pp_s
        
        ratio = np.zeros_like(denominator)
        valid_calc_mask = (denominator != 0) & (self.ells >= 2)
        ratio[valid_calc_mask] = cl_pp_s[valid_calc_mask] / denominator[valid_calc_mask]
        
        cl_pp_res_scaled = cl_pp_s * (1 - ratio)
        cl_pp_res_scaled[self.ells < 2] = 0  # Ensure C_0, C_1 are zero

        cl_phiphi_res_raw = np.zeros(self.lmax_calc + 1)
        # Use self.inv_phi_norm_factor (already handles ells < 2 by being zero there)
        # valid_calc_mask ensures we only operate where ells >= 2 and denom != 0
        cl_phiphi_res_raw[valid_calc_mask] = cl_pp_res_scaled[valid_calc_mask] * self.inv_phi_norm_factor[valid_calc_mask]
        
        cl_phiphi_res_padded = np.zeros(self.lmax_spectra + 1)
        len_to_copy = min(len(cl_phiphi_res_raw), self.lmax_spectra + 1)
        cl_phiphi_res_padded[:len_to_copy] = cl_phiphi_res_raw[:len_to_copy]
        
        return cl_phiphi_res_padded

    def calculate_delensed_cl_bb(self, cl_phiphi_res_padded):
        r"""
        Calculates the delensed B-mode power spectrum.

        Parameters
        ----------
        cl_phiphi_res_padded : numpy.ndarray
            Residual C_ell^phiphi (unscaled), padded to self.lmax_spectra.

        Returns
        -------
        numpy.ndarray
            Delensed C_ell^BB in muK^2, up to self.lmax_spectra.
        """
        pars_delensed = self._get_camb_pars() 
        pars_delensed.set_for_lmax(lmax=self.lmax_spectra, lens_potential_accuracy=0)
        pars_delensed.set_lens_potential_cls(cl_phiphi_res_padded) 

        results_delensed = camb.get_results(pars_delensed)
        powers_delensed_raw = results_delensed.get_cmb_power_spectra(
            pars_delensed, CMB_unit='muK', raw_cl=True)
        cl_bb_delensed = powers_delensed_raw['lensed_scalar'][:, 2]
        
        return cl_bb_delensed

    def calculate_delensing_efficiency(self, cl_bb_lensed, cl_bb_delensed):
        r"""
        Calculates delensing efficiency: 100 * (ClBB_lensed - ClBB_delensed) / ClBB_lensed.

        Parameters
        ----------
        cl_bb_lensed : numpy.ndarray
            Lensed C_ell^BB.
        cl_bb_delensed : numpy.ndarray
            Delensed C_ell^BB.

        Returns
        -------
        numpy.ndarray
            Delensing efficiency in percent.
        """
        min_len = min(len(cl_bb_lensed), len(cl_bb_delensed))
        efficiency = np.zeros(min_len)
        ells_spec = np.arange(min_len)

        # Avoid division by zero and calculate for l >= 2
        mask_eff = (cl_bb_lensed[:min_len] != 0) & (ells_spec >= 2)
        
        delta_bb = cl_bb_lensed[mask_eff] - cl_bb_delensed[mask_eff]
        efficiency[mask_eff] = 100 * delta_bb / cl_bb_lensed[mask_eff]
        
        return efficiency

    def run_calculation(self, output_lmax=100):
        r"""
        Runs the full delensing efficiency calculation pipeline.

        Parameters
        ----------
        output_lmax : int
            Maximum multipole moment for which to save the efficiency.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns 'l' and 'delensing_efficiency'.
        """
        print("Setting up CAMB parameters...")
        pars = self._get_camb_pars()
        print("Running initial CAMB calculation for lensed spectra...")
        results_lensed = camb.get_results(pars)

        print("1. Calculating lensed B-mode power spectrum C_ell^BB...")
        cl_bb_lensed = self.calculate_lensed_cl_bb(pars, results_lensed)

        print("2. Calculating CMB lensing potential power spectrum C_ell^phiphi (scaled)...")
        cl_phiphi_scaled = self.calculate_cl_phiphi_scaled_from_results(results_lensed)

        print("3. Loading lensing noise power spectrum N_ell^phiphi (scaled)...")
        nl_phiphi_scaled = self.load_noise_nl_phiphi()

        print("4. Calculating residual lensing potential power spectrum C_ell^phiphi_res (unscaled)...")
        cl_phiphi_res_padded = self.calculate_residual_cl_phiphi(cl_phiphi_scaled, nl_phiphi_scaled)
        
        print("5. (Padding of residual potential done in step 4)")

        print("6. Calculating delensed B-mode power spectrum C_ell^BB_delensed...")
        cl_bb_delensed = self.calculate_delensed_cl_bb(cl_phiphi_res_padded)

        print("7. Calculating delensing efficiency...")
        delensing_efficiency = self.calculate_delensing_efficiency(cl_bb_lensed, cl_bb_delensed)

        ls_output = np.arange(2, output_lmax + 1)
        # Ensure delensing_efficiency array is long enough
        if len(delensing_efficiency) < output_lmax + 1:
            raise ValueError("Calculated efficiency array is shorter than requested output_lmax.")
        efficiency_output = delensing_efficiency[2 : output_lmax + 1]
        
        output_df = pd.DataFrame({'l': ls_output, 'delensing_efficiency': efficiency_output})
        
        output_filename = "data/result.csv"
        output_df.to_csv(output_filename, index=False, float_format='%.6e')
        print("Results saved to " + str(output_filename))
        
        print("\nDelensing Efficiency Summary:")
        print("l_multipole | Efficiency (%)")
        print("-----------------------------")
        for l_val in [2, 10, 20, 50, 100]:
            if l_val <= output_lmax:
                idx = np.where(output_df['l'] == l_val)[0]
                if len(idx) > 0:
                    eff = output_df['delensing_efficiency'].iloc[idx[0]]
                    print(str(l_val).ljust(11) + " | " + ("%.4f" % eff))
        
        return output_df


def main():
    r"""Main function to run the CMB delensing calculation."""
    
    H0 = 67.5
    ombh2 = 0.022
    omch2 = 0.122
    mnu = 0.06
    omk = 0
    tau = 0.06
    As = 2e-9
    ns = 0.965
    
    noise_file = '/Users/antoidicherianlonappan/Workspace/Benchmarks/data/N0.csv'

    calculator = CMBDelensingCalculator(
        H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau,
        As=As, ns=ns, noise_file_path=noise_file
    )
    
    _ = calculator.run_calculation(output_lmax=100)


if __name__ == '__main__':
    try:
        main()
    except NameError as e:
        if 'camb' in str(e).lower():
            print("Execution failed: CAMB module not found or not imported correctly.")
            print("Please ensure CAMB is installed and accessible in your Python environment.")
        else:
            # Re-raise other NameErrors if they are not related to CAMB
            print("A NameError occurred: " + str(e))
            import traceback
            traceback.print_exc()
    except FileNotFoundError as e:
        print("Execution failed: A required file was not found.")
        print(e)
    except Exception as e:
        print("An unexpected error occurred during execution: " + str(e))
        import traceback
        traceback.print_exc()
