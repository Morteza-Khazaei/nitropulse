import numpy as np
from .utils.fresnel import ReflTransm_PlanarBoundary
from .utils.util import toDB, toPower


class S2RTR:

    def __init__(self, frq_GHz, theta_i, theta_s, phi_i, phi_s, s, cl, eps2, eps3, a, kappa_e, d, acftype, RT_models):
        """
        Initializes the S2RTR class with the given parameters.

        Parameters:
            f : float
                Frequency in GHz.
            theta_i : float
                Incidence angle in degrees.
            theta_s : float
                Scattering angle in degrees.
            phi_i : float
                Azimuth angle of the incident wave in degrees.
            phi_s : float
                Azimuth angle of the scattered wave in degrees.
            s : float
                RMS height of the ground surface (m).
            cl : float
                Correlation length of the surface (m).
            eps2 : complex
                Dielectric constant of the canopy layer.
            eps3 : complex
                Dielectric constant of the ground surface.
            a : float
                Single-scattering albedo (0 < a < 0.2).
            kappa_e : float
                Extinction coefficient of the Rayleigh layer (Np/m).
            d : float
                Thickness of the Rayleigh layer (m).
            acftype : str
                Type of autocorrelation function ('Gaussian', 'Exponential', 'Spherical').
            RT_s : str
                Surface scattering model ('AIEM', 'PRISM1').
            RT_c : str
                Radiative transfer model ('Diff', 'Spec').
            get_sig_ground : bool
                If True, returns the backscatter coefficients of the ground surface.
        """
        
        self.f = frq_GHz
        self.theta_i = theta_i
        self.theta_s = theta_s
        self.phi_i = phi_i
        self.phi_s = phi_s
        self.s = s
        self.cl = cl
        # Air dielectric constant
        self.eps = 1. 
        self.eps2 = eps2
        self.eps3 = eps3
        self.eps_ratio = eps3 / eps2
        self.a = a
        self.kappa_e = kappa_e
        self.d = d
        self.acftype = acftype
        self.RT_s = RT_models['RT_s']  # Surface scattering model
        self.RT_c = RT_models['RT_c']  # Radiative transfer model

        
        self.__check__()


    def __check__(self):
        """
        Checks if the input parameters are valid.
        
        Raises:
            ValueError: If any of the parameters are invalid.
        """
        assert self.a < 0.2, "albedo must be < 0.2"
        assert self.s < 0.5, "RMS height must be < 0.5"
        assert self.cl > 0, "Correlation length must be > 0"
        assert self.kappa_e > 0, "Extinction coefficient must be > 0"
        assert self.d > 0, "Thickness of the Rayleigh layer must be > 0"
        assert self.RT_s in ['AIEM', 'PRISM1', 'Dubois95', 'SMART'], "RT_s must be 'AIEM' or 'PRISM1'"
        assert self.RT_c in ['Diff', 'Spec'], "RT_c must be 'Diff' or 'Spec'"
        assert self.acftype in ['gauss', 'exp', 'pow'], "acftype must be 'Gaussian', 'Exponential', or 'Power-law 1.5'"
        assert self.theta_i >= 0, "Incidence angle must be >= 0"
        assert self.theta_i <= 90, "Incidence angle must be <= 90"
        assert self.theta_s >= 0, "Scattering angle must be >= 0"
        assert self.theta_s <= 90, "Scattering angle must be <= 90"
        assert self.phi_i >= 0, "Azimuth angle of the incident wave must be >= 0"
        assert self.phi_i <= 360, "Azimuth angle of the incident wave must be <= 360"
        assert self.phi_s >= 0, "Azimuth angle of the scattered wave must be >= 0"
        assert self.phi_s <= 360, "Azimuth angle of the scattered wave must be <= 360"
        assert self.f > 0, "Frequency must be > 0"
        assert self.s > 0, "RMS height must be > 0"
        assert self.cl > 0, "Correlation length must be > 0"


    def calc_sigma(self, todB=True):
        """
        Computes the backscatter coefficients for the given parameters.
        
        Returns:
            dict: A dictionary containing the backscatter coefficients in dB for 
                  'vv', 'hh', 'hv', and 'vh' polarizations.
        """

        pol_list = ['vv', 'hh', 'hv', 'vh']
        
        # --- Call the radiative transfer model ---
        if self.RT_c == 'Diff':
            sig_s = {}
            # --- Call the AIEM model ---
            if self.RT_s == 'AIEM':
                from aiem import AIEM0

                # aiem0 = AIEM0(
                #     frq_GHz=self.f, theta_i=self.theta_i, theta_s=self.theta_s, phi_i=self.phi_i, phi_s=self.phi_s, 
                #     s=self.s, l=self.cl, eps=self.eps3, acf_type=self.acftype)
                # sig_s_full = aiem0.run(todB=False).tolist()[0]
                # sig_s = dict(zip(pol_list, sig_s_full))
                
                aiem0 = AIEM0(
                    frq_GHz=self.f, acf=self.acftype, s=self.s, l=self.cl, 
                    thi_deg=self.theta_i, ths_deg=self.theta_s, phi_deg=self.phi_i, phs_deg=self.phi_s, eps=self.eps3)
                
                # Run the AIEM model
                # Note: todB=False to get the results in Power
                sig_s = aiem0.compute_sigma0(pol='full', todB=False)
            
            # --- Call the PRISM1 model ---
            elif self.RT_s == 'PRISM1':
                from .surface.prism1 import PRISM1
                prism0 = PRISM1(f=self.f, theta_i=self.theta_i, eps=self.eps3, s=self.s)
                sig_s_full = prism0.calc_sigma(todB=False)
                # Convert to a dictionary with polarizations
                sig_s = dict(zip(pol_list, sig_s_full))
            elif self.RT_s == 'Dubois95':
                from .surface.dubois95 import Dubois95
                db95 = Dubois95(fGHz=self.f, theta=self.theta_i, eps=self.eps3, s=self.s)
                sig_s_full = db95.calc_sigma(todB=False)
                # Convert to a dictionary with polarizations
                sig_s = dict(zip(pol_list, sig_s_full))
            elif self.RT_s == 'SMART':
                from .surface.smart import SMART
                smart = SMART(fGHz=self.f, theta_deg=self.theta_i, s=self.s, eps=self.eps3)
                sig_s_full = smart.calc_sigma(todB=False)
                # Convert to a dictionary with polarizations
                sig_s = dict(zip(pol_list, sig_s_full))
            else:
                raise ValueError("RT_s must be 'AIEM' or 'PRISM1'")
            
            # --- Call the Rayleigh model ---
            sig_t = self.__S2RTR_DiffuseUB_FullPol(sig_s, self.eps2, self.a, self.kappa_e, self.d, self.theta_i, todB=todB)

            if todB:
                # Convert to dB
                sig_s_dB = {pq: toDB(sig_s[pq]) for pq in pol_list}
                return sig_s_dB, sig_s_dB, sig_t
            else:
                return sig_s, sig_s, sig_t

        elif self.RT_c == 'Spec':
            
            sig_0_top = {}
            sig_0_bot = {}
            
            # --- Call the AIEM model ---
            if self.RT_s == 'AIEM':
                from aiem import AIEM0
                # --- Call the AIEM model for the Rayleigh layer ---
                aiem0 = AIEM0(
                    frq_GHz=self.f, acf=self.acftype, s=self.s, l=self.cl,
                    thi_deg=self.theta_i, ths_deg=self.theta_s, phi_deg=self.phi_i, phs_deg=self.phi_s, eps=self.eps2)
                # Run the AIEM model for the Rayleigh layer
                # Note: todB=False to get the results in Power
                sig_0_top = aiem0.compute_sigma0(pol='full', todB=False)
                
                # --- Call the AIEM model for the ground surface ---
                aiem1 = AIEM0(
                    frq_GHz=self.f, acf=self.acftype, s=self.s, l=self.cl,
                    thi_deg=self.theta_i, ths_deg=self.theta_s, phi_deg=self.phi_i, phs_deg=self.phi_s, eps=self.eps_ratio)
                # Run the AIEM model for the ground surface
                sig_0_bot = aiem1.compute_sigma0(pol='full', todB=False)
                # Note: todB=False to get the results in Power
            
            # --- Call the PRISM1 model ---
            elif self.RT_s == 'PRISM1':
                
                
                # --- Import the PRISM1 model ---
                from .surface.prism1 import PRISM1
                # --- Call the PRISM1 model for the Rayleigh layer ---
                prism0 = PRISM1(f=self.f, theta_i=self.theta_i, eps=self.eps2, s=self.s)
                sig_0_top_full = prism0.calc_sigma(todB=False)
                sig_0_top = dict(zip(pol_list, sig_0_top_full))
                
                # --- Call the PRISM1 model for the ground surface ---
                prism1 = PRISM1(f=self.f, theta_i=self.theta_i, eps=self.eps_ratio, s=self.s)
                sig_0_bot_full = prism1.calc_sigma(todB=False)
                sig_0_bot = dict(zip(pol_list, sig_0_bot_full))
            
            elif self.RT_s == 'Dubois95':

                # --- Import the Dubois95 model ---
                from .surface.dubois95 import Dubois95
                # --- Call the Dubois95 model for the Rayleigh layer ---
                db95_top = Dubois95(fGHz=self.f, theta=self.theta_i, eps=self.eps2, s=self.s)
                sig_0_top_full = db95_top.calc_sigma(todB=False)
                sig_0_top = dict(zip(pol_list, sig_0_top_full))
                
                # --- Call the Dubois95 model for the ground surface ---
                db95_bot = Dubois95(fGHz=self.f, theta=self.theta_i, eps=self.eps_ratio, s=self.s)
                sig_0_bot_full = db95_bot.calc_sigma(todB=False)
                sig_0_bot = dict(zip(pol_list, sig_0_bot_full))
            
            elif self.RT_s == 'SMART':
                from .surface.smart import SMART
                # --- Call the SMART model for the Rayleigh layer ---
                smart_top = SMART(fGHz=self.f, theta_deg=self.theta_i, s=self.s, eps=self.eps2)
                sig_0_top_full = smart_top.calc_sigma(todB=False)
                sig_0_top = dict(zip(pol_list, sig_0_top_full))
                
                # --- Call the SMART model for the ground surface ---
                smart_bot = SMART(fGHz=self.f, theta_deg=self.theta_i, s=self.s, eps=self.eps_ratio)
                sig_0_bot_full = smart_bot.calc_sigma(todB=False)
                sig_0_bot = dict(zip(pol_list, sig_0_bot_full))
            else:
                raise ValueError("RT_s must be 'AIEM' or 'PRISM1'")
            
            # --- Call the Rayleigh model ---
            sig_t = self.__S2RTR_SpecularUB_FullPol(
                sig_0_top, sig_0_bot, self.eps,self.eps2, self.eps3, self.a, self.kappa_e, self.d, self.theta_i, todB=todB)
            
            if todB:
                # Convert to dB
                sig_12_dB = {pq: toDB(sig_0_top[pq]) for pq in pol_list}
                sig_23_dB = {pq: toDB(sig_0_bot[pq]) for pq in pol_list}
                return sig_12_dB, sig_23_dB, sig_t
            else:
                return sig_0_top, sig_0_bot, sig_t
        else:
            raise ValueError("RT_c must be 'Diff' or 'Spec'")


    def __S2RTR_DiffuseUB_FullPol(self, sig_s, eps, a, kappa_e, d, theta_i, todB=True):
        """
        Computes σ₀ for all polarizations (vv, hh, hv, vh) for a weakly
        scattering Rayleigh layer with a diffuse upper boundary.
        
        Parameters:
            sig_s : dict
                Dictionary of σ₀ in Power for:
                    'vv', 'hh', 'hv', 'vh'
            eps : complex
                Complex dielectric constant of the ground surface.
            a : float
                Single-scattering albedo (0 < a < 0.2).
            kappa_e : float
                Extinction coefficient of the Rayleigh layer (Np/m).
            d : float
                Thickness of the Rayleigh layer (m).
            theta_i : float
                Incidence angle in degrees.
        
        Returns:
            A dictionary of σ₀ in dB for:
                'vv', 'hh', 'hv', 'vh'
        """
        theta_rad = np.radians(theta_i)
        cos_theta = np.cos(theta_rad)
        Upsilon = np.exp(-kappa_e * d / cos_theta)
        U2 = Upsilon ** 2
        kappa_s = a * kappa_e

        # --- Reflectivity (Fresnel) ---
        _, _, gammah, gammav, *_ = ReflTransm_PlanarBoundary(1, eps, theta_i)
        Gamma = {
            'v': gammav,
            'h': gammah
        }

        # Initialize output
        sigma_0_db = {}

        # Loop over all polarizations
        for pq in ['vv', 'hh', 'hv', 'vh']:
            p, q = pq[0], pq[1]
            Gamma_p = Gamma[p]
            Gamma_q = Gamma[q]

            # Volume backscatter term
            if pq in ['hv', 'vh']:
                sigma_v_back = a   # You can use e.g., a/2 if needed
                n = 1              # Incoherent
            else:
                sigma_v_back = 3 * a / 4
                n = 2              # Coherent for hh/vv

            sigma_v_bist = kappa_s

            # Apply generalized equation
            sigma_0_lin = (
                U2 * sig_s[pq]
                + sigma_v_back * cos_theta / (2 * kappa_e) * (1 - U2) * (1 + Gamma_p * Gamma_q * U2)
                + n * sigma_v_bist * d * (Gamma_p + Gamma_q) * U2
            )
            
            # Convert to dB
            if todB:
                sigma_0_db[pq] = toDB(sigma_0_lin)
            else:
                sigma_0_db[pq] = sigma_0_lin

        return sigma_0_db


    def __S2RTR_SpecularUB_FullPol(self, sig_0_top, sig_0_bot, eps, eps2, eps3, a, kappa_e, d, theta_i, todB=True):
        """
        Computes sigma_0 for all polarizations (vv, hh, hv, vh) for a weakly scattering
        Rayleigh layer with distinct upper boundary using PRISM models.

        Parameters:
            sig_0_top : dict
                Dictionary of σ₀ in Power for:
                    'vv', 'hh', 'hv', 'vh'
            sig_0_bot : dict
                Dictionary of σ₀ in Power for:
                    'vv', 'hh', 'hv', 'vh'
            eps2 : complex
                Complex dielectric constant of the Rayleigh layer.
            eps3 : complex
                Complex dielectric constant of the ground surface.
            a : float
                Single-scattering albedo (0 < a < 0.2).
            kappa_e : float
                Extinction coefficient of the Rayleigh layer (Np/m).
            d : float
                Thickness of the Rayleigh layer (m).
            theta_i : float
                Incidence angle in degrees.
            todB : bool
                If True, returns the results in dB. If False, returns in Power.

        Returns:
            Dictionary with σ₀ in dB for polarizations: 'vv', 'hh', 'hv', 'vh'.
        """
        theta_rad = np.radians(theta_i)
        sin_theta = np.sin(theta_rad)

        # Transmission angle inside Rayleigh layer
        thetapr_rad = np.arcsin(np.sqrt(1 / eps2.real) * sin_theta)
        thetapr_deg = np.degrees(thetapr_rad)
        costhetapr = np.sqrt(1 - (1 / eps2.real) * sin_theta**2)

        # Scattering coefficient
        kappa_s = a * kappa_e

        # Transmissivity
        T = np.exp(-kappa_e * d / costhetapr)
        T2 = T ** 2

        # Reflectivity and transmissivity of top boundary (air -> layer)
        _, _, _, _, _, _, Th_12, Tv_12 = ReflTransm_PlanarBoundary(eps, eps2, theta_i)

        # Reflectivity of bottom boundary (layer -> ground)
        _, _, gammah_23, gammav_23, *_ = ReflTransm_PlanarBoundary(eps2, eps3, thetapr_deg)
        Gamma = {'v': gammav_23, 'h': gammah_23}
        T_top = {'v': Tv_12, 'h': Th_12}

        # Compute σ₀ for all polarization combinations
        sigma_0 = {}

        for pq in ['vv', 'hh', 'hv', 'vh']:
            p, q = pq[0], pq[1]
            Gamma_p = Gamma[p]
            Gamma_q = Gamma[q]
            Tpq = T_top[p] * T_top[q]
            sig_surface = sig_0_top[pq]
            sig_bottom = sig_0_bot[pq]

            # Backscatter and bistatic coefficients
            if pq in ['hv', 'vh']:
                sigma_v_back = a
                n = 1            # Incoherent
            else:
                sigma_v_back = 3 * a / 4
                n = 2            # Coherent for hh/vv

            sigma_v_bist = kappa_s

            # Generalized version of Eq. 11.79
            sigma_lin = (
                Tpq * (
                    T2 * sig_bottom +
                    sigma_v_back * costhetapr / (kappa_e + kappa_e) * (1 - T2) * (1 + Gamma_p * Gamma_q * T2) +
                    n * sigma_v_bist * d * (Gamma_p + Gamma_q) * T2
                ) + sig_surface
            )
            # Convert to dB
            if todB:
                sigma_0[pq] = toDB(sigma_lin)
            else:
                sigma_0[pq] = sigma_lin

        return sigma_0