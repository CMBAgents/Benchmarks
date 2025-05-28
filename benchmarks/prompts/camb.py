import pandas as pd
import os
from benchmarks import DATA_DIR, Logger

__version__ = 'v1'
filename = os.path.join(DATA_DIR, f'camb_prompts_{__version__}.csv')

logger = Logger("prompt.camb", verbose=True)

def make_prompts():
    df = pd.DataFrame(columns=['statement', 'prompt','reference_code'])
    ################### Problem 1 ####################
    statement = "Compute the CMB Temperature power spectrum for Planck 2018 cosmological parameters with Omega baryon = 0.02."
    prompt = """Calculate the Cosmic Microwave Background (CMB) temperature power spectrum for a flat Lambda CDM cosmology using the following parameters with CAMB:
    Hubble constant ($H_0$): 67.5 km/s/Mpc
    Baryon density ($\\Omega_b h^2$): 0.02
    Cold dark matter density ($\\Omega_c h^2$): 0.122
    Neutrino mass sum ($\\Sigma m_\nu$): 0.06 eV
    Curvature ($\\Omega_k$): 0 (flat cosmology)
    Optical depth to reionization ($\\tau$): 0.06
    Scalar amplitude ($A_s$): $2 \\times 10^{-9}$
    Scalar spectral index ($n_s$): 0.965

    Compute the temperature power spectrum ($l(l+1)C_l^{TT}/(2\\pi)$) in units of $\\mu K^2$ for multipole moments from $l=2$ to $l=3000$. Save the results in a CSV file named result.csv with two columns:
    - l: Multipole moment (integer values from 2 to 3000)
    - TT: Temperature power spectrum ($l(l+1)C_l^{TT}/(2\\pi)$ in $\\mu K^2$)"""
    reference_code = """import camb
import numpy as np
def get_tt():
    H0=67.5
    omch2=0.122
    mnu=0.06
    omk=0
    ombh2=0.02
    tau=0.06
    lmax=3000
    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau,lmax=lmax+500)
    pars.InitPower.set_params(As=2e-9, ns=0.965,)
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    spectra = powers['total']
    return np.arange(lmax+1)[2:], spectra[2:lmax+1,0]
    """

    df.loc[len(df)] = [statement, prompt, reference_code]

    ################### Problem 2 ####################
    statement = "Compute the CMB Temperature power spectrum for Planck 2018 cosmological parameters with Omega k(curvature) = 0.05"
    prompt ="""Calculate the Cosmic Microwave Background (CMB) temperature power spectrum for a non-flat Lambda CDM cosmology using the following parameters with CAMB:
    Hubble constant ($H_0$): 67.3 km/s/Mpc
    Baryon density ($\\Omega_b h^2$): 0.022
    Cold dark matter density ($\\Omega_c h^2$): 0.122
    Neutrino mass sum ($\\Sigma m_\nu$): 0.06 eV
    Curvature ($\\Omega_k$): 0.05
    Optical depth to reionization ($\\tau$): 0.06
    Scalar amplitude ($A_s$): $2 \\times 10^{-9}$
    Scalar spectral index ($n_s$): 0.965

    Compute the temperature power spectrum ($l(l+1)C_l^{TT}/(2\\pi)$) in units of $\\mu K^2$ for multipole moments from $l=2$ to $l=3000$. Save the results in a CSV file named result.csv with two columns:
    l: Multipole moment (integer values from 2 to 3000)
    TT: Temperature power spectrum ($l(l+1)C_l^{TT}/(2\\pi)$ in $\\mu K^2$) """
    reference_code = """import camb
import numpy as np
def get_tt():
    H0=67.5
    omch2=0.122
    mnu=0.06
    ombh2=0.022
    omk=0.05
    tau=0.06
    lmax=3000
    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau,lmax=lmax+500)
    pars.InitPower.set_params(As=2e-9, ns=0.965,)
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    spectra = powers['total']
    return np.arange(lmax+1)[2:], spectra[2:lmax+1,0]
    """
    df.loc[len(df)] = [statement, prompt, reference_code]

    ################### Problem 3 ####################
    statement = "Compute the CMB Temperature power spectrum for Planck 2018 cosmological parameters with H0=70"
    prompt = """Calculate the raw Cosmic Microwave Background (CMB) temperature power spectrum for a flat Lambda CDM cosmology using the following parameters with CAMB:
    Hubble constant ($H_0$): 70 km/s/Mpc
    Baryon density ($\\Omega_b h^2$): 0.022
    Cold dark matter density ($\\Omega_c h^2$): 0.122
    Neutrino mass sum ($\\Sigma m_\nu$): 0.06 eV
    Curvature ($\\Omega_k$): 0
    Optical depth to reionization ($\\tau$): 0.06
    Scalar amplitude ($A_s$): $2 \\times 10^{-9}$
    Scalar spectral index ($n_s$): 0.965

    Compute the temperature power spectrum ($C_l^{TT}$) in units of $\\mu K^2$ for multipole moments from $l=2$ to $l=3000$. Save the results in a CSV file named result.csv with two columns:
    l: Multipole moment (integer values from 2 to 3000)
    TT: Temperature power spectrum ($C_l^{TT}$ in $\\mu K^2$) """
    reference_code = """import camb
import numpy as np
def get_tt():
    H0=70
    omk=0
    omch2=0.122
    mnu=0.06
    ombh2=0.022
    tau=0.06
    lmax=3000
    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau,lmax=lmax+500)
    pars.InitPower.set_params(As=2e-9, ns=0.965,)
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars,raw_cl=True,CMB_unit='muK')
    spectra = powers['total']
    return np.arange(lmax+1)[2:], spectra[2:lmax+1,0]
    """
    df.loc[len(df)] = [statement, prompt, reference_code]

    ################### Problem 4 ####################
    statement = "Compute the CMB Temperature power spectrum for Planck 2018 cosmological parameters with H0=74"
    prompt ="""Calculate the Cosmic Microwave Background (CMB) raw temperature power spectrum for a flat Lambda CDM cosmology using the following parameters with CAMB:
    Hubble constant ($H_0$): 74 km/s/Mpc
    Baryon density ($\\Omega_b h^2$): 0.022
    Cold dark matter density ($\\Omega_c h^2$): 0.122
    Neutrino mass sum ($\\Sigma m_\nu$): 0.06 eV
    Curvature ($\\Omega_k$): 0
    Optical depth to reionization ($\\tau$): 0.06
    Scalar amplitude ($A_s$): $2 \\times 10^{-9}$
    Scalar spectral index ($n_s$): 0.965

    Compute the temperature power spectrum ($C_l^{TT}$) in units of $\\mu K^2$ for multipole moments from $l=2$ to $l=3000$. Save the results in a CSV file named result.csv with two columns:
    l: Multipole moment (integer values from 2 to 3000)
    TT: Raw temperature power spectrum"""

    reference_code = """import camb
import numpy as np
def get_tt():
    H0=74
    omk=0
    omch2=0.122
    mnu=0.06
    ombh2=0.022
    tau=0.06
    lmax=3000
    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau,lmax=lmax+500)
    pars.InitPower.set_params(As=2e-9, ns=0.965,)
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, raw_cl=True, CMB_unit='muK')
    spectra = powers['total']
    return np.arange(lmax+1)[2:], spectra[2:lmax+1,0]
    """
    df.loc[len(df)] = [statement, prompt, reference_code]

    ################### Problem 5 ####################
    statement ="Compute the CMB E-mode power spectrum for Planck 2018 cosmological parameters with optical depth tau = 0.04"
    prompt ="""Calculate the Cosmic Microwave Background (CMB) E-mode polarization power spectrum ($l(l+1)C_l^{EE}/(2\\pi)$) for a flat Lambda CDM cosmology using the following parameters with CAMB:
    Hubble constant ($H_0$): 67.5 km/s/Mpc
    Baryon density ($\\Omega_b h^2$): 0.022
    Cold dark matter density ($\\Omega_c h^2$): 0.122
    Neutrino mass sum ($\\Sigma m_\nu$): 0.06 eV
    Curvature ($\\Omega_k$): 0
    Optical depth to reionization ($\\tau$): 0.04
    Scalar amplitude ($A_s$): $2 \\times 10^{-9}$
    Scalar spectral index ($n_s$): 0.965

    Compute the E-mode power spectrum ($l(l+1)C_l^{EE}/(2\\pi)$) in units of $\\mu K^2$ for multipole moments from $l=2$ to $l=3000$. Save the results in a CSV file named result.csv with two columns:
    l: Multipole moment (integer values from 2 to 3000)
    EE: E-mode polarization power spectrum ($l(l+1)C_l^{EE}/(2\\pi)$ in $\\mu K^2$) """

    reference_code ="""import camb
import numpy as np
def get_ee():
    omk=0
    omch2=0.122
    mnu=0.06
    ombh2=0.022
    H0 = 67.5
    tau=0.04
    lmax=3000
    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau,lmax=lmax+500)
    pars.InitPower.set_params(As=2e-9, ns=0.965,)
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    spectra = powers['total']
    return np.arange(lmax+1)[2:], spectra[2:lmax+1,1]
    """
    df.loc[len(df)] = [statement, prompt, reference_code]

    ################### Problem 6 ####################
    statement ="Compute the total CMB B-mode power spectrum for spectrum for Planck 2018 cosmological parameters with r = 0"
    prompt ="""Calculate the Cosmic Microwave Background (CMB) B-mode polarization power spectrum ($l(l+1)C_l^{BB}/(2\\pi)$) for a flat Lambda CDM cosmology using the following parameters with CAMB:
    Hubble constant ($H_0$): 67.5 km/s/Mpc
    Baryon density ($\\Omega_b h^2$): 0.022
    Cold dark matter density ($\\Omega_c h^2$): 0.122
    Neutrino mass sum ($\\Sigma m_\nu$): 0.06 eV
    Curvature ($\\Omega_k$): 0
    Optical depth to reionization ($\\tau$): 0.06
    Tensor-to-scalar ratio ($r$): 0
    Scalar amplitude ($A_s$): $2 \\times 10^{-9}$
    Scalar spectral index ($n_s$): 0.965

    Compute the B-mode power spectrum ($l(l+1)C_l^{BB}/(2\\pi)$) in units of $\\mu K^2$ for multipole moments from $l=2$ to $l=3000$. Save the results in a CSV file named result.csv with two columns:
    l: Multipole moment (integer values from 2 to 3000)
    BB: B-mode polarization power spectrum ($l(l+1)C_l^{BB}/(2\\pi)$ in $\\mu K^2$) """

    reference_code ="""import camb
import numpy as np
def get_bb():
    omk=0
    omch2=0.122
    mnu=0.06
    ombh2=0.022
    H0 = 67.5
    tau = 0.06
    lmax=3000
    r = 0
    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau, r=r,lmax=lmax+500, lens_potential_accuracy=1)
    pars.WantTensors = False
    pars.InitPower.set_params(As=2e-9, ns=0.965)
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    spectra = powers['total']
    return np.arange(lmax+1)[2:], spectra[2:lmax+1,2]
    """
    df.loc[len(df)] = [statement, prompt, reference_code]

    ################### Problem 7 ####################
    statement ="Compute the total raw CMB B-mode power spectrum for spectrum for Planck 2018 cosmological parameters with r = 0.1"
    prompt ="""Calculate the Cosmic Microwave Background (CMB) raw B-mode polarization power spectrum ($C_l^{BB}$) for a flat Lambda CDM cosmology using the following parameters with CAMB:
    Hubble constant ($H_0$): 67.5 km/s/Mpc
    Baryon density ($\\Omega_b h^2$): 0.022
    Cold dark matter density ($\\Omega_c h^2$): 0.122
    Neutrino mass sum ($\\Sigma m_\nu$): 0.06 eV
    Curvature ($\\Omega_k$): 0
    Optical depth to reionization ($\\tau$): 0.06
    Tensor-to-scalar ratio ($r$): 0.1
    Scalar amplitude ($A_s$): $2 \\times 10^{-9}$
    Scalar spectral index ($n_s$): 0.965

    Compute the B-mode power spectrum ($C_l^{BB}$) in units of $\\mu K^2$ for multipole moments from $l=2$ to $l=3000$. Save the results in a CSV file named result.csv with two columns:
    l: Multipole moment (integer values from 2 to 3000)
    BB: B-mode polarization power spectrum ($C_l^{BB}$ in $\\mu K^2$) """
    reference_code ="""import camb
import numpy as np
def get_bb():
    omk=0
    omch2=0.122
    mnu=0.06
    ombh2=0.022
    H0 = 67.5
    tau = 0.06
    lmax=3000
    r = 0.1
    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau,lmax=lmax+500)
    pars.WantTensors = True
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=r)
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, raw_cl=True, CMB_unit='muK')
    spectra = powers['total']
    return np.arange(lmax+1)[2:], spectra[2:lmax+1,2]
    """

    df.loc[len(df)] = [statement, prompt, reference_code]

    ################### Problem 8 ####################
    statement ="Compute the angular diameter distance for Planck 2018 cosmological parameters between redshift 0 to 4."
    prompt ="""Calculate the angular diameter distance ($d_A$) from redshift $z=0$ to $z=4$ for a flat Lambda CDM cosmology using the following parameter with CAMB:
    Hubble constant ($H_0$): 67.5 km/s/Mpc
    Baryon density ($\\Omega_b h^2$): 0.022
    Cold dark matter density ($\\Omega_c h^2$): 0.122
    Neutrino mass sum ($\\Sigma m_\nu$): 0.06 eV
    Curvature ($\\Omega_k$): 0
    Optical depth to reionization ($\\tau$): 0.06
    Scalar amplitude ($A_s$): $2 \\times 10^{-9}$
    Scalar spectral index ($n_s$): 0.965

    Compute the angular diameter distance ($d_A$) in units of Mpc for 100 evenly spaced redshift points from $z=0$ to $z=4$ using np.linspace(0, 4, 100). Save the results in a CSV file named result.csv with two columns:
    z: Redshift (100 evenly spaced values from 0 to 4)
    d_A: Angular diameter distance (in Mpc) """
    reference_code ="""import camb
import numpy as np
def get_angular_diameter_distance():
    zmin=0
    zmax=4
    z = np.linspace(zmin,zmax,100)
    omk=0
    omch2=0.122
    mnu=0.06
    ombh2=0.022
    H0 = 67.5
    tau = 0.06
    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=2e-9, ns=0.965)
    results = camb.get_background(pars)
    DA = results.angular_diameter_distance(z)
    return z,DA
    """
    df.loc[len(df)] = [statement, prompt, reference_code]

    ################### Problem 9 ####################
    statement = "Compute the linear matter power spectrum with Planck 2018 cosmological parameters for redshift 0"
    prompt ="""Calculate the linear matter power spectrum ($P(k)$) at redshift $z=0$ for a flat Lambda CDM cosmology using the following parameters with CAMB:
    Hubble constant ($H_0$): 67.5 km/s/Mpc
    Baryon density ($\\Omega_b h^2$): 0.022
    Cold dark matter density ($\\Omega_c h^2$): 0.122
    Neutrino mass sum ($\\Sigma m_\nu$): 0.06 eV
    Curvature ($\\Omega_k$): 0
    Optical depth to reionization ($\\tau$): 0.06
    Scalar amplitude ($A_s$): $2 \\times 10^{-9}$
    Scalar spectral index ($n_s$): 0.965
    k maximum ($k_{max}$): 2

    Compute the linear matter power spectrum('get_matter_power_spectrum') ($P(k)$) in units of (Mpc/$h$)$^3$ for 200 evenly spaced $k$ values in the range $10^{-4} < kh < 1$ (Mpc$^{-1}$). Save the results in a CSV file named result.csv with two columns:
    kh: Wavenumber (in $h$/Mpc, 200 evenly spaced values)
    P_k: Linear matter power spectrum (in (Mpc/$h$)$^3$) """

    reference_code ="""import camb
import numpy as np
def get_matter_power_spectrum():
    z = 0
    omk=0
    omch2=0.122
    mnu=0.06
    ombh2=0.022
    H0 = 67.5
    tau = 0.06
    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=2e-9, ns=0.965)
    pars.set_matter_power(redshifts=[z], kmax=2.0)
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)
    return kh, pk[0]
    """
    df.loc[len(df)] = [statement, prompt, reference_code]

    ################### Problem 10 ####################
    statement ="Compute the delensed total raw CMB B-mode power spectrum with Planck 2018 cosmological parameters for a delensing efficiency of 10%."
    prompt ="""Calculate the raw delensed Cosmic Microwave Background (CMB) B-mode polarization power spectrum ($C_\\ell^{BB}$) for a flat Lambda CDM cosmology using the following parameters with CAMB:
    Hubble constant ($H_0$): 67.5 km/s/Mpc
    Baryon density ($\\Omega_b h^2$): 0.022
    Cold dark matter density ($\\Omega_c h^2$): 0.122
    Neutrino mass sum ($\\Sigma m_\nu$): 0.06 eV
    Curvature ($\\Omega_k$): 0
    Optical depth to reionization ($\\tau$): 0.06
    Tensor-to-scalar ratio ($r$): 0.1
    Scalar amplitude ($A_s$): $2 \\times 10^{-9}$
    Scalar spectral index ($n_s$): 0.965

    Compute the delensed B-mode power spectrum ($C_\\ell^{BB}$) in units of $\\mu K^2$ for multipole moments from $l=2$ to $l=3000$, applying a delensing efficiency of 10%. Save the results in a CSV file named result.csv with two columns:
    l: Multipole moment (integer values from 2 to 3000)
    BB: Delensed B-mode polarization power spectrum ($C_\\ell^{BB}$ in $\\mu K^2$) """
    reference_code ="""import camb
import numpy as np
def get_bb_delensed():
    r = 0.1
    delens_eff = 10
    omk=0
    omch2=0.122
    mnu=0.06
    ombh2=0.022
    H0 = 67.5
    tau = 0.06
    lmax=3000
    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau,lmax=lmax+500)
    pars.WantTensors = True
    pars.InitPower.set_params(As=2e-9, ns=0.965,r=r)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars,raw_cl=True, CMB_unit='muK')
    spectra_tensor = powers['tensor']
    Alens = 1 - (delens_eff/100)
    spectra =results.get_partially_lensed_cls(Alens,CMB_unit='muK',raw_cl=True,lmax=lmax+500)
    return np.arange(lmax+1)[2:], spectra[2:lmax+1,2] + spectra_tensor[2:lmax+1,2]
    """
    df.loc[len(df)] = [statement, prompt, reference_code]

    ################### Problem 11 ####################
    statement ="Compute the delensed CMB Temperature power spectrum with Planck 2018 cosmological parameters for a delensing efficiency of 80%."
    prompt ="""Calculate the delensed Cosmic Microwave Background (CMB) temperature power spectrum ($l(l+1)C_l^{TT}/(2\\pi)$) for a flat Lambda CDM cosmology using the following parameters with CAMB:
    Hubble constant ($H_0$): 67.5 km/s/Mpc
    Baryon density ($\\Omega_b h^2$): 0.022
    Cold dark matter density ($\\Omega_c h^2$): 0.122
    Neutrino mass sum ($\\Sigma m_\nu$): 0.06 eV
    Curvature ($\\Omega_k$): 0
    Optical depth to reionization ($\\tau$): 0.06
    Scalar amplitude ($A_s$): $2 \\times 10^{-9}$
    Scalar spectral index ($n_s$): 0.965

    Compute the delensed temperature power spectrum ($l(l+1)C_l^{TT}/(2\\pi)$) in units of $\\mu K^2$ for multipole moments from $l=2$ to $l=3000$, applying a delensing efficiency of 80%. Save the results in a CSV file named result.csv with two columns:
    l: Multipole moment (integer values from 2 to 3000)
    TT: Delensed temperature power spectrum ($l(l+1)C_l^{TT}/(2\\pi)$ in $\\mu K^2$) """
    reference_code ="""import camb
import numpy as np
def get_tt_delensed():
    omk=0
    omch2=0.122
    mnu=0.06
    ombh2=0.022
    H0 = 67.5
    tau = 0.06
    lmax=3000
    delens_eff = 80
    pars = camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau, lmax=lmax+500)
    pars.InitPower.set_params(As=2e-9, ns=0.965,)
    results = camb.get_results(pars)
    Alens = 1 - (delens_eff/100)
    spectra =results.get_partially_lensed_cls(Alens,CMB_unit='muK',lmax=lmax+500)
    return np.arange(lmax+1)[2:], spectra[2:lmax+1,0]
    """
    df.loc[len(df)] = [statement, prompt, reference_code]

    ################### Problem 12 ####################
    statement ="Compute the CMB E-mode power spectrum for Planck 2018 cosmological parameters with optical depth tau = 0.1 with exponential reionization history."
    prompt = """Calculate the Cosmic Microwave Background (CMB) E-mode polarization power spectrum ($l(l+1)C_l^{EE}/(2\\pi)$) for a flat Lambda CDM cosmology using the following parameters with CAMB:
    Hubble constant ($H_0$): 67.5 km/s/Mpc
    Baryon density ($\\Omega_b h^2$): 0.022
    Cold dark matter density ($\\Omega_c h^2$): 0.122
    Scalar amplitude ($A_s$): $1.8 \\times 10^{-9} \\times e^{2 \\times \\tau}$
    Scalar spectral index ($n_s$): 0.95
    Optical depth to reionization ($\\tau$): 0.1
    Reionization model: Exponential reionization with exponent power $2$

    Compute the E-mode power spectrum ($l(l+1)C_l^{EE}/(2\\pi)$) in units of $\\mu K^2$ for multipole moments from $l=2$ to $l=100$. Save the results in a CSV file named result.csv with two columns:
    l: Multipole moment (integer values from 2 to 100)
    EE: E-mode polarization power spectrum ($l(l+1)C_l^{EE}/(2\\pi)$in $\\mu K^2$)"""
    reference_code ="""import camb
import numpy as np
def get_ee():
    omk=0
    omch2=0.122
    mnu=0.06
    ombh2=0.022
    H0 = 67.5
    lmax=100
    tau = 0.1
    As = 1.8e-9*np.exp(2*tau)
    exp_pow = 2
    pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, As=As, ns=0.95, reionization_model='ExpReionization', tau=tau, reion_exp_power = exp_pow, **{'Reion.timestep_boost':1})
    data= camb.get_results(pars)
    cl = data.get_cmb_power_spectra(pars, lmax=lmax, CMB_unit='muK')['total']
    return np.arange(lmax+1)[2:], cl[2:lmax+1, 1]
    """
    df.loc[len(df)] = [statement, prompt, reference_code]


    ################### Problem 13 ####################
    statement = "Compute the relative difference in the linear matter power spectrum between normal and inverted neutrino hierarchy models at redshift 0."
    prompt ="""Calculate the relative difference in the linear matter power spectrum ($P(k)$) at redshift $z=0$ between two neutrino hierarchy models (normal and inverted) for a flat Lambda CDM cosmology using the following parameters  with CAMB:
    Hubble constant ($H_0$): 67.5 km/s/Mpc
    Baryon density ($\\Omega_b h^2$): 0.022
    Cold dark matter density ($\\Omega_c h^2$): 0.122
    Neutrino mass sum ($\\Sigma m_\nu$): 0.11 eV
    Scalar amplitude ($A_s$): $2 \\times 10^{-9}$
    Scalar spectral index ($n_s$): 0.965

    Compute the linear matter power spectrum using 'get_matter_power_spectrum' ($P(k/h)$) in units of (Mpc/$h$)$^3$ for 200 evenly spaced $k$ values in the range $10^{-4} < k h < 2$ (Mpc$^{-1}$) for both:
    Normal neutrino hierarchy
    Inverted neutrino hierarchy
    
    Calculate the relative difference as $(P(k){\\text{inverted}} / P(k){\\text{normal}} - 1)$. Save the results in a CSV file named result.csv with two columns:
    k: Wavenumber (in $h$/Mpc, 200 evenly spaced values)
    rel_diff: Relative difference in the matter power spectrum ($(P(k){\\text{inverted}} / P(k){\\text{normal}} - 1)$) """
    reference_code = """import camb
def pk_neutrino_comparison():
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.11, neutrino_hierarchy='normal')
    pars.InitPower.set_params(As=2e-9, ns=0.965)
    pars.set_matter_power(redshifts=[0.], kmax=2.0)
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=2, npoints = 200)
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.11, neutrino_hierarchy='inverted')
    pars.InitPower.set_params(As=2e-9, ns=0.965,)
    results = camb.get_results(pars)
    kh2, z2, pk2 = results.get_matter_power_spectrum(minkh=1e-4, maxkh=2, npoints = 200)
    return kh, (pk2[0,:]/pk[0,:]-1)
    """
    df.loc[len(df)] = [statement, prompt, reference_code]

    ################### Problem 14 ####################
    statement = "Compute the delensing efficiency of the CMB B-mode polarization power spectrum for a flat Lambda CDM cosmology given the lensing noise power."
    prompt =f"""Calculate the delensing efficiency of the Cosmic Microwave Background (CMB) B-mode polarization power spectrum for a flat Lambda CDM cosmology using the following parameters with CAMB given the lensing noise power:
    Hubble constant ($H_0$): 67.5 km/s/Mpc
    Baryon density ($\\Omega_b h^2$): 0.022
    Cold dark matter density ($\\Omega_c h^2$): 0.122
    Neutrino mass sum ($\\Sigma m_\nu$): 0.06 eV
    Curvature ($\\Omega_k$): 0
    Optical depth to reionization ($\\tau$): 0.06
    Scalar amplitude ($A_s$): $2 \\times 10^{-9}$
    Scalar spectral index ($n_s$): 0.965

    Load the lensing noise power spectrum ($N_0$) from the file '{os.path.join(DATA_DIR,'N0.csv')}' with columns l and Nl. Here $Nl = N_0 \\times (\\ell(\\ell+1))^2 / (2\\pi)$, and use values up to a maximum multipole $l=2000$. 
    Compute the following::
    1. The lensed B-mode power spectrum ($C_\\ell^{{BB}}$) in units of $\\mu K^2$ for multipole moments up to $l=2000$.
    2. The CMB lensing potential power spectrum ($C_\\ell^{{\\phi\\phi}} (\\ell(\\ell+1))^2 / (2\\pi) $) up to $l=2000$.
    3. Change the lensing potential array and noise power spectrum to have the same length $l=2000$.
    4. The residual lensing potential power spectrum, defined as cl_pp_res = cl_pp*(1 - (cl_pp/(cl_pp+n0))) or $C_\\ell^{{\\phi\\phi}} \\times (1 - (C_\\ell^{{\\phi\\phi}} / (C_\\ell^{{\\phi\\phi}} + N_0)))$ or like cl_pp_res = cl_pp*(1 - (cl_pp/(cl_pp+n0))).
    5. Pad the the residual lensing potential array to have same length as in Params.max_l 
    6. The delensed B-mode power spectrum ($C_\\ell^{{BB,\\text{'delensed'}}}$) using the residual lensing potential power spectrum.
    7. The delensing efficiency as $100 \\times (C_\\ell^{{BB,\\text{'lensed'}}} - C_\\ell^{{BB,\\text{'delensed'}}})/ C_\\ell^{{BB,\\text{'lensed'}}}.

    Save the results in a CSV file named 'result.csv' with two columns:
    l: Multipole moment (integer values from 2 to 100)
    delensing_efficiency: Delensing efficiency (in percent, single value for $l=2$ to $l=100$) """
    reference_code = f"""import camb
import numpy as np
def delensing_efficiency():
    lmax = 2000
    l, n0 = np.loadtxt('{os.path.join(DATA_DIR,'N0.csv')}',delimiter=',', skiprows=1).T
    l = l[:lmax+1]
    n0 = n0[:lmax+1]
    pars = camb.set_params(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06, As=2e-9, ns=0.965,lmax=2000)
    pars.InitPower.set_params(As=2e-9, ns=0.965)
    results = camb.get_results(pars)
    bb_lens = results.get_cmb_power_spectra(pars, CMB_unit='muK')['lensed_scalar'][:lmax+1,2]
    cl_pp =results.get_cmb_power_spectra(pars,CMB_unit='muK')['lens_potential'][:lmax+1,0]
    cl_pp_res = cl_pp*(1 - (cl_pp/(cl_pp+n0)))
    cl_pp_res = np.pad(cl_pp_res, (0, pars.max_l+1-len(cl_pp_res)), mode='constant')
    results = camb.get_results(pars)
    bb_delens = results.get_lensed_cls_with_spectrum(cl_pp_res,CMB_unit='muK')[:lmax+1,2]
    delens_eff = 100 *  (bb_lens[2:101]-bb_delens[2:101])/bb_lens[2:101]
    return np.arange(len(delens_eff))+2, delens_eff"""
    df.loc[len(df)] = [statement, prompt, reference_code]
    return df

def get_prompts(reintial=False):
    if os.path.isfile(filename) and not reintial:
        logger.log(f"Loading prompts from file({__version__}):{filename}",'info' )
        df = pd.read_csv(filename)
    else:
        logger.log(f"""Prompts file not found, creating the dataframe.
                     Version: {__version__}
                     Saving to file: {filename}""", 'info')
            
        df = make_prompts()
        df.to_csv(filename,index=False)
    return df