
import os
import json
import pandas as pd
from tqdm.auto import tqdm
from bisect import bisect_right

from nitropulse.risma import RismaData
from nitropulse.radar import S1Data



class BBCH:
    """
    class to calculate BBCH stages based on cumulative soil GDD and SST.
    """
    def __init__(self, workspace_dir):
        """
        Initialize the Inverse class.
        """
        risma = RismaData(workspace_dir)
        self.df_risma = risma.load_df()

        s1 = S1Data(workspace_dir)
        self.df_S1 = s1.load_df()

        # Load lc_base_temp from the JSON file
        crop_base_temp_file = os.path.join(workspace_dir, 'config', 'gdd', 'crop_base_temp.json')
        with open(crop_base_temp_file, 'r') as f:
            crop_base_temp = json.load(f)
        self.crop_base_temp = crop_base_temp

        # Load crop_gdd_thresh from the JSON file
        gdd_file = os.path.join(workspace_dir, 'config', 'gdd', 'crop_gdd_thresh.json')
        with open(gdd_file, 'r') as f:
            crop_gdd_thresholds = json.load(f)
        self.crop_gdd_thresholds = crop_gdd_thresholds


    def run(self):
        pheno_df = self.get_pheno_df()
        return pheno_df

    def get_pheno_df(self):
        
        # Time-based merge: align S1 (UTC) with nearest RISMA timestamp (UTC) per Station
        # Ensure datetime columns
        self.df_risma = self.df_risma.sort_values('date')
        self.df_S1 = self.df_S1.sort_values('date')

        # Merge asof on time and Station within 2 hours tolerance
        merged_df = pd.merge_asof(
            self.df_S1,
            self.df_risma,
            on='date',
            by='Station',
            tolerance=pd.Timedelta('2h'),
            direction='nearest'
        )

        # Recompute year and doy from the merged timestamp to avoid column clashes
        merged_df['year'] = merged_df['date'].dt.year
        merged_df['doy'] = merged_df['date'].dt.dayofyear
        # Drop any leftover suffix columns from inputs
        for col in ['year_x', 'year_y', 'doy_x', 'doy_y']:
            if col in merged_df.columns:
                merged_df.drop(columns=[col], inplace=True)
        
        # drop nodata inplace based on VV and VH
        merged_df.dropna(subset=['VV', 'VH', 'angle'], inplace=True)

        # Create a new column 'BASE_TEMP' in merged_df
        merged_df['BASE_TEMP'] = merged_df['lc'].map(self.crop_base_temp)

        #  Use interploate to fill Nan in SST
        merged_df['SST'] = merged_df['SST'].interpolate()

        # Calculate cumulative GDD based solely on soil temperature (preferred)
        merged_df['cum_GDD'] = (
            (merged_df['mean_sst'] - merged_df['BASE_TEMP'])
            .clip(lower=0)
            .groupby(merged_df['year'])
            .cumsum()
        )
        # Clean up any legacy columns if present
        for col in ['cum_GDD_air', 'cum_GDD_soil']:
            if col in merged_df.columns:
                merged_df.drop(columns=[col], inplace=True)

        tqdm.pandas(desc="Calculating BBCH")
        # Calculate BBCH stage based on cumulative GDD and soil temperature
        merged_df['BBCH'] = merged_df.progress_apply(lambda row: self.get_bbch_from_soil_gdd(row['lc'], row['cum_GDD'], row['SST']), axis=1)

        # Calculate the cumulative sum of SSM for each year
        merged_df['cum_SSM'] = merged_df.groupby('year')['SSM'].cumsum()

        return merged_df
    

    def get_bbch_from_soil_gdd(self, crop: str, cum_gdd: float, sst: float) -> int:
        """
        Map cumulative soil GDD to a BBCH stage via piecewise linear interpolation
        between defined thresholds.
        """
        # crop = crop.lower()
        if crop not in self.crop_gdd_thresholds:
            # Unknown landcover class: skip BBCH computation
            return None
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

    
