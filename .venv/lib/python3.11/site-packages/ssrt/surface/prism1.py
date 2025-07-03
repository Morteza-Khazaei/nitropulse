import numpy as np
from ..utils.fresnel import Fresn_Refl0, Fresn_Refl
from ..utils.util import toLambda, toPower, toDB

class PRISM1:
    
    def __init__(self, f, theta_i, eps, s):
        """
        Computes sigma_0 for all three polarization combinations 
        based on the PRISM-1 forward model.

        Parameters:
            f : float
                Frequency in GHz.
            theta_i : float
                Incidence angle in degrees.
            eps : complex
                Complex dielectric constant of the surface.
            s : float
                Surface roughness parameter (standard deviation of surface height).

        Returns:
            sig_0_vv : float
                Sigma_0 for VV polarization (dB).
            sig_0_hh : float
                Sigma_0 for HH polarization (dB).
            sig_0_hv : float
                Sigma_0 for HV polarization (dB).
        """
        self.theta_rad = np.radians(theta_i)
        self.eps = eps
        lambda_m = toLambda(f)  # Wavelength in meters
        k = 2 * np.pi / lambda_m  # Wavenumber
        self.ks = k * s  # Roughness parameter
    

    def calc_sigma(self, todB=True):

        gamma0 = Fresn_Refl0(self.eps)       # Normal incidence reflectivity
        gammav, gammah = Fresn_Refl(self.eps, self.theta_rad)  # Angular-dependent reflectivity

        p = (1 - (2 * self.theta_rad / np.pi) ** (1 / (3 * gamma0)) * np.exp(-self.ks)) ** 2
        q = 0.23 * np.sqrt(gamma0) * (1 - np.exp(-self.ks))
        g = 0.70 * (1 - np.exp(-0.65 * self.ks ** 1.8))

        cos_theta = np.cos(self.theta_rad)
        sigvv = g * (cos_theta ** 3) / np.sqrt(p) * (gammav + gammah)

        sig_0_vv = sigvv
        sig_0_hh = sigvv * p
        sig_0_hv = sigvv * q
        sig_0_vh = sigvv * q

        # Convert to dB
        if todB:
            sig_0_vv = toDB(sig_0_vv)
            sig_0_hh = toDB(sig_0_hh)
            sig_0_hv = toDB(sig_0_hv)
            sig_0_vh = toDB(sig_0_vh)

        return sig_0_vv, sig_0_hh, sig_0_hv, sig_0_vh