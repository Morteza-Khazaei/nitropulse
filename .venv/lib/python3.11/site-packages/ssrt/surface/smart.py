import numpy as np
import matplotlib.pyplot as plt

class SMART:
    """
    SMART Forward Model for radar backscatter estimation.

    Attributes:
        frequency (float): Frequency in GHz
        dielectric_constant (complex): Complex dielectric constant (ε = ε' - jε'')
        theta_deg (float): Incidence angle in degrees
        rms_height_cm (float): Surface RMS height in cm

    Methods:
        compute(): Computes sigma_0_vv and sigma_0_hh (in dB)
    """

    def __init__(self, fGHz, theta_deg, s, eps):
        self.eps = eps
        self.theta_deg = theta_deg
        self.s = s * 100  # Convert to cm
        self.f = fGHz

        # Outputs (in dB)
        self.sigma_0_vv = None
        self.sigma_0_hh = None

    def calc_sigma(self, todB=True):
        """Compute sigma_0_vv and sigma_0_hh in dB using the SMART forward model."""
        eps_pr = np.real(self.eps)
        theta_rad = np.radians(self.theta_deg)
        wavelength_cm = 30.0 / self.f  # Convert GHz to cm
        ks = self.s * (2 * np.pi / wavelength_cm)
        # print(f'ks: {ks}')

        # sigma_0_hh
        sig0_hh = (
            10 ** (-2.75)
            * np.cos(theta_rad) ** 1.5
            / np.sin(theta_rad) ** 5
            * wavelength_cm ** 0.7
            * (ks * np.sin(theta_rad)) ** 1.4
            * 10 ** (0.028 * eps_pr * np.tan(theta_rad))
        )

        # sigma_0_vv
        sig0_vv = (
            10 ** (-2.35)
            * np.cos(theta_rad) ** 3
            / np.sin(theta_rad) ** 3
            * wavelength_cm ** 0.7
            * (ks * np.sin(theta_rad)) ** 1.1
            * 10 ** (0.046 * eps_pr * np.tan(theta_rad))
        )

        # Convert to dB
        if todB:
            sig0_hh = 10 * np.log10(sig0_hh)
            sig0_vv = 10 * np.log10(sig0_vv)
        else:
            sig0_hh = np.asarray(sig0_hh)
            sig0_vv = np.asarray(sig0_vv)
        
        return sig0_vv, sig0_hh, 1e-9, 1e-9  # hv and vh are not calculated in SMART, set to a small value


    def plot(self):
        f = plt.figure()
        ax = f.add_subplot(111)
        t = np.rad2deg(self.theta_deg)
        ax.plot(t, self.sigma_0_hh, color='blue', label='hh')
        ax.plot(t, self.sigma_0_vv, color='red', label='vv')
       # ax.plot(t, 10.*np.log10(self.hv), color='green', label='hv')
        ax.grid()
        # ax.set_ylim(-35.,-5.)
        # ax.set_xlim(30.,70.)
        ax.legend()
        ax.set_xlabel('incidence angle [deg]')
        ax.set_ylabel('backscatter [dB]')


if __name__ == "__main__":
    # Example usage:
    sm = SMART(5.405, 40, 0.1, 5+1*1j)
    sm.compute()
    print(sm.get_results())