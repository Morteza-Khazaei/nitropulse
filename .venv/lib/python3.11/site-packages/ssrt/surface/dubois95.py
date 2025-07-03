"""
implements the Dubois95 model
as described in Ulaby (2014), Chapter 10.6
"""
import numpy as np
import matplotlib.pyplot as plt


class Dubois95(object):
    def __init__(self, fGHz, theta, eps, s):
        """
        Parameters
        ----------
        lam : float
            wavelength in meter
        """
        self.eps = eps
        
        self.theta = np.deg2rad(theta)

        c0 = 299792458  # Speed of light in vacuum (m/s)

        # Frequency and wavenumber
        fHz = fGHz * 1e9  # Hz
        self.lam = (c0 / fHz) * 1e2  # cm

        k = 2 * np.pi / self.lam  # Wavenumber (rad/cm)
        self.ks = k * s  * 1e2  # cm
        # print(f'ks: {self.ks} from backscatter model')


    def todB(self, x):
        return 10.*np.log10(x)
    
    def calc_sigma(self, todB=True):
        lam = self.lam
        ks = self.ks  # ks has no unity (s [cm or m], k [1/cm or 1/m])
        vv = self._vv(lam, ks)
        hh = self._hh(lam, ks)
        hv = 1e-9  # hv is not calculated in Dubois95, set to a small value
        vh = 1e-9  # vh is not calculated in Dubois95, set to a small value
        
        if todB:
            vv = self.todB(vv)
            hh = self.todB(hh)
            hv = self.todB(hv)
            vh = self.todB(vh)
        else:
            vv = np.asarray(vv)
            hh = np.asarray(hh)
            hv = np.asarray(hv)
            vh = np.asarray(vh)
        
        return vv, hh, hv, vh

    def _hh(self, lam, ks):
        """
        lam : float
            wavelength in cm
        """

        a = (10.**-2.75)*(np.cos(self.theta)**1.5)/(np.sin(self.theta)**5.)
        c = 10.**(0.028*np.real(self.eps)*np.tan(self.theta))
        d = ((ks*np.sin(self.theta))**1.4)*lam**0.7

        return a*c*d

    def _vv(self, lam, ks):
        """ eq. 10.41b """
        b = 10.**(-2.35)*((np.cos(self.theta)**3.) / (np.sin(self.theta)**3.))
        c = 10.**(0.046*np.real(self.eps)*np.tan(self.theta))
        d = (ks*np.sin(self.theta))**1.1*lam**0.7

        return b*c*d

    def plot(self):
        f = plt.figure()
        ax = f.add_subplot(111)
        t = np.rad2deg(self.theta)
        ax.plot(t, self.hh, color='blue', label='hh')
        ax.plot(t, self.vv, color='red', label='vv')
       # ax.plot(t, 10.*np.log10(self.hv), color='green', label='hv')
        ax.grid()
        # ax.set_ylim(-35.,-5.)
        # ax.set_xlim(30.,70.)
        ax.legend()
        ax.set_xlabel('incidence angle [deg]')
        ax.set_ylabel('backscatter [dB]')



if __name__ == '__main__':
    # example
    db95 = Dubois95(fGHz=5.405, theta=np.arange(30, 75, 5), eps=5+1*1j, s=0.01)
    print(db95.vv)
    print(db95.hh)
    # db95.plot()
    # plt.show()