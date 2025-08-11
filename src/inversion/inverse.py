import os
import json
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

from aiem import AIEM0
from ssrt import S2RTR
from ssrt.surface import PRISM1, SMART
from ssrt.utils import Dobson85




class Inverse:
    """
    Class to handle inversion data processing.
    """

    def __init__(self, workspace_dir, fGHz, models, acftype='exp'):
        """
        Initialize the Inverse class with directories and crop GDD thresholds.
        """
        self.fGHz = fGHz
        self.models = models
        self.acftype = acftype

        # Load it back as a dict
        crop_bbch_k_b_coff_file = os.path.join(workspace_dir, 'config', 'gdd', 'crop_bbch_k_b_coff.json')
        with open(crop_bbch_k_b_coff_file, "r") as f:
            crop_bbch_k_b_coff = json.load(f)
        self.crop_bbch_k_b_coff = crop_bbch_k_b_coff
        
        # Load crop bounds from the JSON file
        crop_bounds_file = os.path.join(workspace_dir, 'config', 'inversion', 'crop_inversion_bounds.json')
        with open(crop_bounds_file, "r") as f:
            crop_inversion_bounds = json.load(f)
        self.crop_inversion_bounds = crop_inversion_bounds


    def run(self, df):
        # Implementation of the run method will go here
        df = df.groupby(['op', 'year', 'doy', 'angle'], group_keys=False).apply(lambda x: self.inversion(x, self.crop_inversion_bounds))

        # drop na
        df.dropna(inplace=True)

        # drop rows with vvs lower than -50
        df = df[df.vvs > -50]

        return df


    def to_power(self, dB):
        return 10**(dB/10)

    def to_dB(self, power):
        return 10*np.log10(power)
    
    def estimate_crop_height_interp(self, croptype, BBCH, RVI):
        """
        Estimate crop height using Equation 2 (PH = k * RVI + b) with k and b interpolated from BBCH data.

        Parameters:
        -----------
        BBCH : float
            BBCH code representing the phenological stage.
        RVI : float
            Radar Vegetation Index.
        bbch_k_b_data : list of tuples
            List of (BBCH, k, b) values derived from Figures 3 and 4, e.g., [(10, 50, 5), (20, 70, 7), ...].

        Returns:
        --------
        float
            Estimated plant height (PH) in centimeters.

        Raises:
        -------
        ValueError
            If bbch_k_b_data has fewer than 2 points for interpolation.
        """

        bbch_k_b_data = self.crop_bbch_k_b_coff[croptype]['bbch_k_b']

        # Ensure sufficient data points for interpolation
        if len(bbch_k_b_data) < 2:
            raise ValueError("bbch_k_b_data must have at least 2 points for interpolation.")

        # Sort data by BBCH to ensure ascending order
        bbch_k_b_data.sort(key=lambda x: x[0])

        # Extract BBCH, k, and b values
        bbch_values = [x[0] for x in bbch_k_b_data]
        k_values = [x[1] for x in bbch_k_b_data]
        b_values = [x[2] for x in bbch_k_b_data]

        # Create interpolation functions for k and b
        k_interp = interp1d(bbch_values, k_values, kind='linear', fill_value='extrapolate')
        b_interp = interp1d(bbch_values, b_values, kind='linear', fill_value='extrapolate')

        # Compute k and b for the given BBCH
        k = k_interp(BBCH)
        b = b_interp(BBCH)

        # Calculate plant height using Equation 2
        ph = (k * RVI + b) / 100

        # Define min and max for PH
        min_ph = ph - 0.001
        if min_ph < 0:
            min_ph = 0
        max_ph = ph + 0.001

        return min_ph, ph, max_ph
    
    def residuals_local(self, params, fGHz, acftype, RT_models, mv, vv_obs, theta_i, rvi, sand, clay, bulk, sst):
        d, c, s, l, omega = params

        ke = c * np.sqrt(rvi)

        db85 = Dobson85(clay=clay, sand=sand, mv=mv, freq=fGHz, temp=sst, bulk=bulk)
        eps2 = eps3 = np.array([db85.eps,], dtype=complex)

        # Create an instance of the S2RTR class
        rt = S2RTR(
            frq_GHz=fGHz, theta_i=theta_i, theta_s=theta_i, phi_i=0., phi_s=179.999,
            s=s, cl=l, eps2=eps2/2., eps3=eps3, a=omega, kappa_e=ke, d=d, acftype=acftype, RT_models=RT_models)
        
        # Calculate backscattering coefficients
        sig_a, sig_s, sig_c = rt.calc_sigma(todB=False)
        vv_sim = sig_s['vv'] + sig_c['vv']

        vv_residual = np.square(vv_obs - vv_sim)

        return vv_residual
    
    def inversion(self, df_x, bounds):

        dvvs = []
        cvvs = []
        wvvs = []
        vv_vegs = []
        vv_soils = []
        SSRs = []
        SSRl = []
        rvis = []
        heights = []

        for idx, row in df_x.iterrows():
            # print(idx)
            vv = self.to_power(row['VV'])
            vh = self.to_power(row['VH'])
            rvi = (4 * vh) / (vh + vv)
            theta_i = row['angle']
            ssm = row['SSM']
            sst = row['SST']
            bbch = row['BBCH']
            croptype = str(int(row['lc']))
            sand = row['Sand']
            silt = row['Silt']
            clay = row['Clay']
            bulk = row['bulk']
            doy = row['doy']
            op = row['op']


            try:
                c_bound = bounds[croptype]['cbound']
                s_bound = bounds[croptype]['sbound']
                l_bound = bounds[croptype]['lbound']
                w_bound = bounds[croptype]['wbound']

                # Example usage
                height = self.estimate_crop_height_interp(croptype, bbch, rvi)

                # Initial guess for mv and ks
                initial_guess = [height[1], c_bound[1], s_bound[1], l_bound[1], w_bound[1]]

                # Perform the optimization
                res = least_squares(self.residuals_local, initial_guess, args=(self.fGHz, self.acftype, self.models, ssm, vv, theta_i, rvi, sand, clay, bulk, sst),
                    bounds=([height[0], c_bound[0], s_bound[0], l_bound[0], w_bound[0]], [height[2], c_bound[2], s_bound[2], l_bound[2], w_bound[2]]))
                d, c, s, l, w = res.x

                db85 = Dobson85(clay=clay, sand=sand, bulk=bulk, mv=ssm, freq=self.fGHz, t=sst)
                eps = np.array([db85.eps,], dtype=complex)

                pol_list = ['vv', 'hh', 'hv', 'vh']

                if self.models['RT_s'] == 'AIEM':

                    aiem_obj = AIEM0(frq_GHz=self.fGHz, acf=self.acftype, s=s, l=l, thi_deg=theta_i, ths_deg=theta_i, phi_deg=0, phs_deg=179.999, eps=eps)

                    # Compute backscattering coefficient for VV polarization
                    sigma0 = aiem_obj.compute_sigma0(pol='vv', todB=False)
                    vv_soil = sigma0['vv'][0]
                    
                elif self.models['RT_s'] == 'PRISM1':
                    prism0 = PRISM1(f=self.fGHz, theta_i=theta_i, eps=eps, s=s)
                    sig_0_top_full = prism0.calc_sigma(todB=False)
                    sig_0_top = dict(zip(pol_list, sig_0_top_full))

                    vv_soil = sig_0_top['vv'][0]
                
                elif self.models['RT_s'] == 'SMART':

                    smart = SMART(fGHz=self.fGHz, theta_deg=theta_i, s=s, eps=eps)
                    sig_0_full = smart.calc_sigma(todB=False)
                    sig_0_top = dict(zip(pol_list, sig_0_full))

                    vv_soil = sig_0_top['vv'][0]

                vv_veg = vv - vv_soil


            except:
                d = np.nan
                c = np.nan
                s = np.nan
                l = np.nan
                w = np.nan
                vv_veg = np.nan
                vv_soil = np.nan
                rvi = np.nan
                height = np.nan

            dvvs.append(d)
            cvvs.append(c)
            wvvs.append(w)
            vv_vegs.append(vv_veg)
            vv_soils.append(vv_soil)
            SSRs.append(s)
            SSRl.append(l)
            rvis.append(rvi)
            heights.append(height[1])

        # update df_x with new values
        df_x['d'] = dvvs
        df_x['c'] = cvvs
        df_x['w'] = wvvs
        df_x['vvv'] = self.to_dB(vv_vegs)
        df_x['vvs'] = self.to_dB(vv_soils)
        df_x['s'] = SSRs
        df_x['l'] = SSRl
        df_x['rvi'] = rvis
        df_x['height'] = heights

        return df_x



