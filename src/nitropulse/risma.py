

import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from tqdm.auto import tqdm
from risma import AquariusWebPortal


class RismaData:
    """
    Class to handle Risma soil data processing.
    """
    def __init__(self, workspace_dir):
        """
        Initialize the RismaData class.
        """
        # Define the local directory path for storing RISMA CSV files.
        self.workspace_dir = workspace_dir
        self.risma_dir = os.path.join(workspace_dir, 'inputs', 'RISMA_CSV_files')
        os.makedirs(self.risma_dir, exist_ok=True)
        print(f"Local directory for RISMA data: {self.risma_dir}")
    
    def download_risma_data(self, out_dir, stations, parameters, sensors, depths, start_date, end_date):
        """
        Download RISMA measurements from data center
        """

        if not out_dir:
            user_home = os.path.expanduser('~')
            out_dir = os.path.join(user_home, 'Downloads', 'RISMA_CSV')
        
        # make dir if not exit
        os.makedirs(out_dir, exist_ok=True)
        
        server="agrifood.aquaticinformatics.net"
        aafc = AquariusWebPortal(server=server, auto_accept_disclaimer=True)

        # Load params
        params = aafc.fetch_params()

        # Filter parameters
        level_params = params[params.param_name.isin(parameters)]

        # Filter stations
        stations = aafc.fetch_locations(stations=stations)

        # Load available datasets
        datasets = aafc.fetch_datasets(
            param_names=level_params.param_name.tolist(), 
            stations=stations.loc_id.tolist(), 
            sensors=sensors, depths=depths)
        
        # groupby datasets based on loc_id
        gp_df = datasets.groupby('loc_id')
        pbar = tqdm(gp_df, desc="Downloading RISMA data")
        for station, df in pbar:
            fname = os.path.join(out_dir, f'{station}_{start_date.split("-")[0]}_to_{end_date.split("-")[0]}.csv')
            pbar.set_description(f"Downloading for {station}")

            # Check if file already exits
            if os.path.exists(fname):
                print(f"The file '{fname}' exists. \n\tSkipping download process for {station}.")
                continue
            else:
                print(f"The file '{fname}' does not exist. \n\tProceeding with download process for {station}.")

                st_df = aafc.fetch_dataset(dset_names=df.dset_name.to_list(), start=start_date, end=end_date, extra_data_types=None)

                # Save the DataFrame to a CSV file named 'data.csv'
                st_df.to_csv(fname, index=False)
        
        return None


    def load_df(self, depth='0 to 5 cm'):
        """        Load the Risma soil data from CSV files in the specified directory.
        This method processes both ascending and descending data from the Risma project.
        It reads CSV files, processes the data, and returns a DataFrame containing the processed data.
        Args:
            risma_dir (str): The directory containing the Risma CSV files.
        Returns:
            pd.DataFrame: A DataFrame containing the processed Risma soil data.
        """

        list_of_dfs = []

        # Loop through each file in the directory
        # file_list = sorted(os.listdir(self.risma_dir), key=lambda x:int(x.split('_')[1][2:]))
        file_list = os.listdir(self.risma_dir)
        print(f"Found {len(file_list)} files in RISMA directory.")
        pbar = tqdm(file_list, desc="Loading RISMA files")
        for filename in pbar:
            if filename.endswith(".csv"):  # Check if the file is a CSV file
                base = os.path.splitext(filename)[0]
                # Expect filenames like 'RISMA_MB8.csv' -> station 'MB8'
                parts = base.split('_')
                station_name = parts[1] if len(parts) > 1 else base
                pbar.set_description(f"Loading {filename}")
                filepath = os.path.join(self.risma_dir, filename)
                print(f"Processing RISMA file: {filepath} for station {station_name}")
                df_RISMA = self.read_risma_bulk_csv(filepath, station_name)

                # Only append non-empty frames to avoid concat errors
                if df_RISMA is not None and not df_RISMA.empty:
                    list_of_dfs.append(df_RISMA)
                else:
                    print(f"No usable soil data found in {filename}; skipping.")

        
        # Concatenate the two dataframes
        # Ensure empty fallback frame has all downstream-required columns
        expected_cols = ['sensor', 'value', 'station', 'depth', 'mean_sst', 'mean_airt']
        # Build safe empty frames if needed
        df_empty = pd.DataFrame(columns=expected_cols)
        df_empty.index.name = 'date'

        df_RISMA = pd.concat(list_of_dfs) if list_of_dfs else df_empty.copy()

        # Reset the index to make 'date' a regular column
        df_RISMA.reset_index(inplace=True)

        # Ensure a depth column exists; default to requested depth label if missing
        if 'depth' not in df_RISMA.columns:
            df_RISMA['depth'] = depth

        # Keep rows with '0 to 5 cm depth'
        df_RISMA = df_RISMA[df_RISMA['depth'] == depth]

        # Ensure mean_airt and mean_sst columns exist before interpolation
        if 'mean_airt' not in df_RISMA.columns:
            df_RISMA['mean_airt'] = pd.NA
        if 'mean_sst' not in df_RISMA.columns:
            df_RISMA['mean_sst'] = pd.NA

        # Interpolate mean_airt and mean_sst
        df_RISMA['mean_airt'] = pd.to_numeric(df_RISMA['mean_airt'], errors='coerce').interpolate()
        df_RISMA['mean_sst'] = pd.to_numeric(df_RISMA['mean_sst'], errors='coerce').interpolate()
        # Ensure station column exists even for empty/edge cases
        if 'station' not in df_RISMA.columns:
            df_RISMA['station'] = pd.NA
        # Vectorized sensor categorization to avoid apply shape issues
        sensor_str = df_RISMA['sensor'].astype(str)
        df_RISMA['category'] = np.select(
            [
                sensor_str.str.contains('Soil water content', na=False),
                sensor_str.str.contains('Soil temperature', na=False),
                sensor_str.str.contains('Air temperature', na=False),
                sensor_str.str.contains('Precipitation totals', na=False),
            ],
            ['SSM', 'SST', 'AIRT', 'PRCP'],
            default='Unknown'
        )

        # pivot the table
        pivoted = df_RISMA.pivot(
            index=['date', 'station', 'depth', 'mean_sst', 'mean_airt'], 
            columns='category', 
            values='value'
        )
        pivoted.reset_index(inplace=True)
        # Ensure 'date' is datetime for downstream .dt accessors
        if 'date' in pivoted.columns:
            pivoted['date'] = pd.to_datetime(pivoted['date'], errors='coerce')
        else:
            pivoted['date'] = pd.to_datetime(pd.Series([], dtype='datetime64[ns]'))

        # add year
        pivoted['year'] = pivoted['date'].dt.year

        # add day of year as doy
        pivoted['doy'] = pivoted['date'].dt.dayofyear

        # Dynamically load texture data for the requested depth label
        def to_depth_cm(depth_label: str) -> int:
            if not isinstance(depth_label, str):
                return 5
            m = re.search(r"(\d+)\s*cm", depth_label)
            if m:
                return int(m.group(1))
            return 5

        requested_depth_cm = to_depth_cm(depth)
        texture_df = self.load_stations_texture(depth=requested_depth_cm)

        # Merge the dataframes based on the station column
        df_RISMA = pd.merge(pivoted, texture_df, on='station', how='left')

        # Normalize column names to lowercase for downstream consistency
        df_RISMA.columns = df_RISMA.columns.map(lambda c: c.lower() if isinstance(c, str) else c)

        return df_RISMA


    # Create new columns 'SSM' and 'SST' based on conditions
    def categorize_sensor(self, row):
        if 'Soil water content' in row['sensor']:
            return 'SSM'
        elif 'Soil temperature' in row['sensor']:
            return 'SST'
        else:
            return 'Unknown'  # Handle cases where neither condition is met
    
    def load_stations_texture(self, depth=5):
        """
        Load the RISMA stations texture data from JSON.
        """
        # Desired workspace config path
        ws_cfg_dir = os.path.join(self.workspace_dir, 'config', 'risma')
        ws_json_path = os.path.join(ws_cfg_dir, 'stations_texture.json')

        # Expect setup_workspace to have copied the packaged default into workspace.
        # If missing, fall back to copying from packaged config.
        if not os.path.exists(ws_json_path):
            pkg_default = os.path.join(os.path.dirname(__file__), 'config', 'risma', 'stations_texture.json')
            if os.path.exists(pkg_default):
                os.makedirs(ws_cfg_dir, exist_ok=True)
                with open(pkg_default, 'r') as fsrc, open(ws_json_path, 'w') as fdst:
                    fdst.write(fsrc.read())
                print(f"Seeded stations texture JSON to {ws_json_path}")
            else:
                raise FileNotFoundError(f"Missing stations texture JSON at {ws_json_path}")

        with open(ws_json_path, 'r') as f:
            rima_stations = json.load(f)

        # Flatten into rows
        all_data = [
            {"station": station, **measurement}
            for station, measurements in rima_stations.items()
            for measurement in measurements
        ]

        df_texture = pd.DataFrame(all_data)

        # Rename columns for better readability
        df_texture = df_texture.rename(columns={
            "depth_cm": "depth_cm",
            "density_gcm3": "bulk",
            "sand_pct": "sand",
            "silt_pct": "silt",
            "clay_pct": "clay",
            "classification": "classification"
        })

        # Keep rows with requested depth (cm)
        df_texture = df_texture[df_texture['depth_cm'] == depth]

        # Normalize soil texture percentages to 0 to 1 range
        df_texture['sand'] = df_texture['sand'] / 100
        df_texture['silt'] = df_texture['silt'] / 100
        df_texture['clay'] = df_texture['clay'] / 100

        return df_texture


    def read_risma_bulk_csv(self, fname, station):

        # Read raw file without trusting header row; RISMA bulk export contains multiple header lines
        raw = pd.read_csv(fname, header=None, dtype=str, low_memory=False)

        # Helper to normalize header text (lowercase, strip, remove BOM)
        def norm(s):
            if pd.isna(s):
                return ''
            return str(s).replace('\ufeff', '').strip().lower()

        # Find cell containing 'Timestamp (UTC)' in any column (use DataFrame.map to avoid applymap deprecation)
        norm_df = raw.map(norm)
        ts_mask_any = norm_df.eq('timestamp (utc)') | norm_df.map(lambda x: ('timestamp' in x) if isinstance(x, str) else False)
        if not ts_mask_any.values.any():
            # If still not found, return empty frame with expected structure
            empty = pd.DataFrame(columns=['sensor', 'value', 'station', 'depth', 'mean_sst', 'mean_airt'])
            empty.index.name = 'date'
            return empty

        # Use the first occurrence
        ts_pos = ts_mask_any.stack().idxmax()  # (row_idx, col_idx)
        ts_row, ts_col = ts_pos
        names_row_idx = max(ts_row - 1, 0)

        # Build column names combining name tokens and units from the identified columns onward
        names_row = raw.iloc[names_row_idx, ts_col:].fillna('')
        units_row = raw.iloc[ts_row, ts_col:].fillna('')

        def build_name(name_token, unit_token):
            # Extract detail after the dot if present (e.g., 'Air Temp.Air temperature' -> 'Air temperature')
            name_token = str(name_token)
            if '.' in name_token:
                name_token = name_token.split('.', 1)[1]
            # Normalize degree symbol encoding
            unit_token = str(unit_token).replace('Ã‚', '').replace('Â°', '°')
            # Replace 'Value' with actual variable name
            if 'Value' in unit_token:
                return unit_token.replace('Value', name_token).strip()
            # If units missing, just use the name token
            return name_token.strip()

        columns = []
        for j in range(len(names_row)):
            if j == 0:
                columns.append('Timestamp (UTC)')
            else:
                columns.append(build_name(names_row.iloc[j], units_row.iloc[j]))

        # Data starts after the units/header row; slice from timestamp column to the end
        data = raw.iloc[ts_row + 1:, ts_col:].reset_index(drop=True)
        data.columns = columns[: data.shape[1]]

        # Parse timestamp and set as index
        data['Timestamp (UTC)'] = pd.to_datetime(data['Timestamp (UTC)'], errors='coerce')
        data = data.dropna(subset=['Timestamp (UTC)'])
        data.set_index('Timestamp (UTC)', inplace=True)

        # Convert the data columns to numeric where possible
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Note: Keep full timestamped series (UTC). We'll match to S1 by time later.

        # calculate mean daily air temperature first group hourly rows into daily rows
        air_candidates = [
            c for c in data.columns
            if re.match(r'^Air\s*temperature', c.replace('Â°', '°'), flags=re.I) or re.match(r'^Air\s*Temp', c, flags=re.I)
        ]
        if air_candidates:
            air_series = data[air_candidates[0]]
        else:
            # Create a NaN series if air temperature is missing to keep pipeline robust
            air_series = pd.Series(index=data.index, dtype=float)
        df_stats_air = air_series.resample('D').agg(['min', 'max'])
        df_stats_air['mean_airt'] = df_stats_air['min'].add(df_stats_air['max']).div(2.)
        df_stats_air.drop(columns=['min', 'max'], inplace=True)

        # Detect precipitation totals (instantaneous or interval values)
        prcp_candidates = [
            c for c in data.columns
            if re.match(r'^Precipitation\s*totals', c, flags=re.I) or re.match(r'^Precip\s*Total', c, flags=re.I)
        ]
        if prcp_candidates:
            prcp_series = data[prcp_candidates[0]]
        else:
            prcp_series = pd.Series(index=data.index, dtype=float)

        # Robustly locate soil columns at 0–5 cm depth
        def match_cols(prefix):
            cols = []
            for c in data.columns:
                c_norm = c.replace('Â°', '°')
                # Accept either '0 to 5 cm' or '5 cm depth'
                depth_ok = re.search(r'(0\s*to\s*5\s*cm|\b5\s*cm\s*depth)', c_norm, flags=re.I)
                if re.search(rf'^{prefix}', c_norm, flags=re.I) and depth_ok:
                    cols.append(c)
            return cols

        ssm_cols = match_cols('Soil water content')
        sst_cols = match_cols('Soil temperature')

        # Compute mean across sensors for each timestamp; keep native timestamps (UTC)
        ssm_series = data[ssm_cols].mean(axis=1) if ssm_cols else pd.Series(index=data.index, dtype=float)
        sst_series = data[sst_cols].mean(axis=1) if sst_cols else pd.Series(index=data.index, dtype=float)

        df_stats_sst = sst_series.resample('D').agg(['min', 'max'])
        df_stats_sst['mean_sst'] = df_stats_sst['min'].add(df_stats_sst['max']).div(2.)
        df_stats_sst.drop(columns=['min', 'max'], inplace=True)

        # Build long format with instantaneous SSM, SST, Air temperature, and Precipitation; keep timestamps
        df_soil = pd.DataFrame({
            'Soil water content': ssm_series,
            'Soil temperature': sst_series,
            'Air temperature': air_series,
            'Precipitation totals': prcp_series,
        })
        # fill nan
        df_soil.ffill(inplace=True)
        df_soil.bfill(inplace=True)

        df_melted = df_soil.melt(ignore_index=False, var_name='sensor', value_name='value')
        df_melted.index.name = 'date'

        # Attach station and constant depth label
        df_melted['station'] = station
        df_melted['depth'] = '0 to 5 cm'

        # Merge daily mean_airt and mean_sst onto the timestamped soil records via day key
        df_melted = df_melted.reset_index()
        df_melted['date_day'] = df_melted['date'].dt.floor('D')
        df_stats_air = df_stats_air.rename_axis('date').reset_index()
        df_stats_sst = df_stats_sst.rename_axis('date').reset_index()
        df_melted = df_melted.merge(df_stats_air, left_on='date_day', right_on='date', how='left', suffixes=('', '_air'))
        df_melted = df_melted.merge(df_stats_sst, left_on='date_day', right_on='date', how='left', suffixes=('', '_sst'))
        df_melted = df_melted.drop(columns=['date_air', 'date_sst', 'date_day'])
        df_melted = df_melted.set_index('date')

        return df_melted


if __name__ == "__main__":
    # Example usage
    workspace_dir = './temp_workspace'
    
    risma_data = RismaData(workspace_dir)
    df_risma = risma_data.load_df()
    
    print(df_risma.head())
    print("Risma data loaded successfully.")
