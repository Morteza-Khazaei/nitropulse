import pandas as pd
from bisect import bisect_right

from inversion.risma import RismaData
from inversion.radar import S1Data



class BBCH:
    """
    class to calculate BBCH stages based on cumulative soil GDD and SST.
    """
    def __init__(self, workspace_dir, risma_dir, s1_dir, crop_gdd_thresholds=None, auto_download=False):
        """
        Initialize the Inverse class.
        """
        risma = RismaData(workspace_dir, risma_dir)
        self.df_risma = risma.load_df()

        s1 = S1Data(workspace_dir, s1_dir, auto_download)
        self.df_S1 = s1.load_df()

        self.crop_gdd_thresholds = crop_gdd_thresholds


    def run(self, lc_base_temp=None):
        
        # Merge 'grouped' and 'df_RISMA_asc' based on 'year' and 'doy'
        merged_df = pd.merge(self.df_risma, self.df_S1, on=['year', 'doy', 'op', 'station'], how='inner')
        
        # drop nodata inplace based on VV and VH
        merged_df.dropna(subset=['VV', 'VH', 'angle'], inplace=True)

        # Create a new column 'BASE_TEMP' in merged_df
        merged_df['BASE_TEMP'] = merged_df['lc'].map(lc_base_temp)

        #  Use interploate to fill Nan in SST
        merged_df['SST'].interpolate(inplace=True)

        # Convert daily soil temperature series to yearly cumulative soil GDD.
        merged_df['cum_GDD_air'] = merged_df.groupby(['year'], group_keys=False).apply(self.compute_cumulative_air_gdd)
        merged_df['cum_GDD_soil'] = merged_df.groupby(['year'], group_keys=False).apply(self.compute_cumulative_soil_gdd)

        # add cum_GDD based on mean of GDD
        merged_df['cum_GDD'] = merged_df[['cum_GDD_air', 'cum_GDD_soil']].mean(axis=1)

        # Calculate BBCH stage based on cumulative GDD and soil temperature
        merged_df['BBCH'] = merged_df.apply(lambda row: self.get_bbch_from_soil_gdd(row['lc'], row['cum_GDD'], row['SST'], row['BASE_TEMP']), axis=1)

        # Calculate the cumulative sum of SSM for each year
        merged_df['cum_SSM'] = merged_df.groupby(['year'], group_keys=False)['SSM'].apply(lambda x: x.cumsum())

        return merged_df
    

    def compute_cumulative_soil_gdd(self, df):
        """Convert daily soil temperature series to cumulative soil GDD."""
        soil_temps = df['mean_sst']
        base_temp = df['BASE_TEMP']
        daily = (soil_temps - base_temp).clip(lower=0)
        return daily.cumsum()

    def compute_cumulative_air_gdd(self, df):
        """Convert daily soil temperature series to cumulative soil GDD."""
        air_temp = df['mean_airt']
        base_temp = df['BASE_TEMP']
        daily = (air_temp - base_temp).clip(lower=0)
        return daily.cumsum()

    def get_bbch_from_soil_gdd(self, crop: str, cum_gdd: float, sst: float, base_temp: float) -> int:
        """
        Map cumulative soil GDD to a BBCH stage via piecewise linear interpolation
        between defined thresholds.
        """
        # crop = crop.lower()
        if crop not in self.crop_gdd_thresholds:
            raise ValueError(f"No thresholds defined for '{crop}'")
        thresh = self.crop_gdd_thresholds[crop]

        # Before first threshold
        if cum_gdd < thresh[0][0]:
            return 0
        # After last threshold
        if cum_gdd >= thresh[-1][0]:
            return thresh[-1][1]

        # Find interval
        idx = bisect_right([g for g, _ in thresh], cum_gdd)

        bbch = None

        # Check if idx is within the valid range
        if idx < len(thresh):
            g0, b0 = thresh[idx - 1]
            g1, b1 = thresh[idx]
            frac = (cum_gdd - g0) / (g1 - g0)
            bbch = round(b0 + frac * (b1 - b0))
        else:
            # Handle the case where idx is out of range (e.g., cum_gdd is greater than the largest threshold)
            bbch = thresh[-1][1]  # Return the last defined BBCH value

        # check if sst is negative return 0 for bbch otherwise return bbch
        if sst < 0:
            return None
        else:
            return bbch