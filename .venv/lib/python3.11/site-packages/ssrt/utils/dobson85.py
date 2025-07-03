import numpy as np

"""
Generic model for dielectric mixing models
"""

class EpsModel(object):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        clay : float
            clay content as fractional volume
        sand : float
            sand content as fractional volume
        bulk : float
            bulk density [g/cm**3]; default: 1.65
        mv : float
            volumetric soil moisture content [cm**3/cm**3] = [g/cm**3]
        freq : float
            frequency [GHz]
        t : float
            temperature [°C]
        """

        self.clay = kwargs.get('clay', None)
        self.sand = kwargs.get('sand', None)

        self.bulk = kwargs.get('bulk', 1.65)

        self.mv = kwargs.get('mv', None)

        self.f = kwargs.get('freq', None)

        self.t = kwargs.get('temp', 23.)

        self._check()

    def _check(self):
        assert self.clay is not None, 'Clay needs to be provided!'
        assert self.clay >=0.
        assert self.clay <=1.
        assert self.sand is not None, 'Sand needs to be provided!'
        assert self.sand >=0.
        assert self.sand <=1.
        assert self.mv is not None, 'volumetric soil moisture needs to be given'
        assert self.f is not None, 'Frequency needs to be given!'
        if isinstance(self.f, np.ndarray):
            assert np.all(self.f> 0.)
        else:
            assert self.f > 0.

"""
Dielectric mixing model for soils
after Dobson et al. (1985)
coding after Ulaby (2014), Chapter 4
"""
class Dobson85(EpsModel):
    def __init__(self, **kwargs):
        super(Dobson85, self).__init__(**kwargs)

        self.debye = kwargs.get('debye', False)
        self.single_debye = kwargs.get('single_debye', False)
        self._init_model_parameters()
        self.ew = self._calc_ew()
        self.eps = self._calc_eps()

    def _calc_ew(self):
        """
        calculate dielectric permittivity of free water
        using either the Debye model or a more simplistic approach
        """
        if self.debye:
            # single Debye dielectric model for pure water. Eqs. 4.14 or Debye model with conductivity term for e2. Eqs. 4.67
            return self._debye()
        else:
            # default setting
            # simplistic approach using Eq. 4.69
            return self._simple_ew()

    def _simple_ew(self):
        """
        eq. 4.69
        simplistic approach with T=23°C, bulk density = 1.7 g/cm3
        """
        f0 = 18.64   # relaxation frequency [GHz]
        hlp = self.f/f0
        e1 = 4.9 + (74.1)/(1.+hlp**2.)
        e2 =(74.1*hlp)/(1.+hlp**2.) + 6.46 * self.sigma/self.f
        return e1 + 1.j * e2

    def _debye(self):
        """
        Debye model
        1) single Debye dielectric model for pure water. Eqs. 4.14
        2) (default) Debye model with conductivity term for e2. Eqs. 4.67
        """

        f = self.f *10**9
        ew_inf = 4.9 # determined by Lane and Saxton 1952 (E.4.15)
        ew_0 = 88.045 - 0.4147 * self.t + 6.295*10**-4 * self.t**2 + 1.075*10**-5 * self.t**3
        tau_w = (1.1109*10**-10 - 3.824*10**-12*self.t + 6.938*10**-14*self.t**2 - 5.096*10**-16*self.t**3)/2./np.pi
        e1 = ew_inf +(ew_0-ew_inf)/(1 + (2*np.pi*f*tau_w)**2)

        if self.single_debye:
            # single Debye dielectric model for pure water. Eqs. 4.14
            e2 = 2*np.pi*f*tau_w * (ew_0-ew_inf) / (1 + (2*np.pi*f*tau_w)**2)
        else:
            # Debye model with conductivity term for e2. Eqs. 4.67
            e2 = 2*np.pi*f*tau_w * (ew_0-ew_inf) / (1 + (2*np.pi*f*tau_w)**2) + (2.65-self.bulk)/2.65/self.mv * self.sigma/(2*np.pi*8.854*10**-12*f)
        return e1 + 1.j *e2

    def _init_model_parameters(self):
        """
        model parameters, eq. 4.68, Ulaby (2014)
        """
        self.alpha = 0.65
        self.beta1 = 1.27-0.519*self.sand - 0.152*self.clay
        self.beta2 = 2.06 - 0.928*self.sand -0.255*self.clay
        self.sigma = -1.645 + 1.939*self.bulk - 2.256*self.sand + 1.594*self.clay

    def _calc_eps(self):
        """
        calculate dielectric permittivity
        Eq. 4.66 (Ulaby et al., 2014)
        """

        e1 = (1.+0.66*self.bulk+self.mv**self.beta1*np.real(self.ew)**self.alpha - self.mv)**(1./self.alpha)
        e2 = np.imag(self.ew)*self.mv**self.beta2
        return e1 + 1.j*e2