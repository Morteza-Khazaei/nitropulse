import numpy as np

def Fresn_Refl0(eps):
    """
    Fresnel reflectivity at normal incidence.
    
    Parameters:
        eps : complex
            Complex dielectric constant.
    
    Returns:
        gamma0 : float
            Reflectivity (unitless).
    """
    sqrt_eps = np.sqrt(eps)
    gamma0 = np.abs((1 - sqrt_eps) / (1 + sqrt_eps)) ** 2
    return gamma0


def Fresn_Refl(eps, theta_rad):
    """
    Fresnel reflectivities for vertical and horizontal polarization.

    Parameters:
        eps : complex
            Complex dielectric constant.
        theta_rad : float
            Incidence angle in radians.

    Returns:
        gammav : float
            Vertical polarization reflectivity.
        gammah : float
            Horizontal polarization reflectivity.
    """
    rho_v, rho_h = refl_coef(theta_rad, 1, eps)
    gammav = np.abs(rho_v) ** 2
    gammah = np.abs(rho_h) ** 2
    return gammav, gammah


def refl_coef(theta_rad, eps1, eps2):
    """
    Computes vertical and horizontal polarized reflection coefficients
    of a plane dielectric surface.

    Parameters:
        theta_rad : float
            Incidence angle in radians.
        eps1 : float
            Dielectric constant of incident medium.
        eps2 : complex
            Dielectric constant of transmission medium.

    Returns:
        rho_v : complex
            Vertical reflection coefficient.
        rho_h : complex
            Horizontal reflection coefficient.
    """
    n1 = np.sqrt(eps1)
    n2 = np.sqrt(eps2)
    sin_theta = np.sin(theta_rad)
    sin_theta_ratio = (n1 * sin_theta) / n2
    costh2 = np.sqrt(1 - sin_theta_ratio ** 2)

    rho_v = -(n2 * np.cos(theta_rad) - n1 * costh2) / (n2 * np.cos(theta_rad) + n1 * costh2)
    rho_h =  (n1 * np.cos(theta_rad) - n2 * costh2) / (n1 * np.cos(theta_rad) + n2 * costh2)

    return rho_v, rho_h

def ReflTransm_PlanarBoundary(eps1, eps2, theta1d):
    """
    Computes the reflection and transmission coefficients, reflectivities, 
    and transmissivities at a planar boundary for both horizontal (h) and 
    vertical (v) polarizations.

    Parameters:
        eps1 : complex
            Relative dielectric constant of medium 1.
        eps2 : complex
            Relative dielectric constant of medium 2.
        theta1d : float
            Incidence angle in medium 1 (degrees).

    Returns:
        rhoh : complex
            Reflection coefficient for horizontal polarization.
        rhov : complex
            Reflection coefficient for vertical polarization.
        gammah : float
            Reflectivity for horizontal polarization.
        gammav : float
            Reflectivity for vertical polarization.
        tauh : complex
            Transmission coefficient for horizontal polarization.
        tauv : complex
            Transmission coefficient for vertical polarization.
        Th : float
            Transmissivity for horizontal polarization.
        Tv : float
            Transmissivity for vertical polarization.
    """
    theta1 = np.radians(theta1d)

    sqrt_eps1 = np.sqrt(eps1)
    sqrt_eps2 = np.sqrt(eps2)

    sin_theta2 = sqrt_eps1 / sqrt_eps2 * np.sin(theta1)
    cos_theta2 = np.sqrt(1 - sin_theta2 ** 2)

    rhoh = (sqrt_eps1 * np.cos(theta1) - sqrt_eps2 * cos_theta2) / \
           (sqrt_eps1 * np.cos(theta1) + sqrt_eps2 * cos_theta2)
    
    rhov = (sqrt_eps1 * cos_theta2 - sqrt_eps2 * np.cos(theta1)) / \
           (sqrt_eps1 * cos_theta2 + sqrt_eps2 * np.cos(theta1))

    tauh = 1 + rhoh
    tauv = (1 + rhov) * (np.cos(theta1) / cos_theta2)

    gammah = np.abs(rhoh) ** 2
    gammav = np.abs(rhov) ** 2

    Th = 1 - gammah
    Tv = 1 - gammav

    return rhoh, rhov, gammah, gammav, tauh, tauv, Th, Tv