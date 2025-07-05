

import os
import re
import pandas as pd
from datetime import datetime, timedelta


class RismaData:
    """
    Class to handle Risma soil data processing.
    """
    def __init__(self, workspace_dir, risma_dir, ):
        """
        Initialize the RismaData class.
        """
        self.risma_dir = os.path.join(workspace_dir, risma_dir)
        self.df_texture = self.load_stations_texture(depth=5)
    
    def download_risma_data(self,):
        """
        Download the Risma soil data from the specified directory.
        This method is a placeholder for downloading data if needed.
        """
        print("Downloading Risma data is not implemented yet.")
        pass
    
    def load_df(self, depth='0 to 5 cm'):
        """        Load the Risma soil data from CSV files in the specified directory.
        This method processes both ascending and descending data from the Risma project.
        It reads CSV files, processes the data, and returns a DataFrame containing the processed data.
        Args:
            risma_dir (str): The directory containing the Risma CSV files.
        Returns:
            pd.DataFrame: A DataFrame containing the processed Risma soil data.
        """

        list_of_dfs_risma_asc = []
        list_of_dfs_risma_desc = []

        # Loop through each file in the directory
        for filename in os.listdir(self.risma_dir):
            if filename.endswith(".csv"):  # Check if the file is a CSV file
                station_name = filename.split('_')[1]
                filepath = os.path.join(self.risma_dir, filename)
                print(f"Processing file: {filename}")
                df_RISMA_asc = self.read_risma_bulk_csv(filepath, station_name, S1_lot='18:30')
                df_RISMA_desc = self.read_risma_bulk_csv(filepath, station_name, S1_lot='06:30')

                list_of_dfs_risma_asc.append(df_RISMA_asc)
                list_of_dfs_risma_desc.append(df_RISMA_desc)

        
        # Concatenate the two dataframes
        df_RISMA_asc = pd.concat(list_of_dfs_risma_asc)
        df_RISMA_desc = pd.concat(list_of_dfs_risma_desc)

        # Reset the index to make 'date' a regular column
        df_RISMA_asc.reset_index(inplace=True)
        df_RISMA_desc.reset_index(inplace=True)

        # add orbit pass
        df_RISMA_asc['op'] = 0
        df_RISMA_desc['op'] = 1

        # Concatenate the two dataframes
        df_RISMA = pd.concat([df_RISMA_asc, df_RISMA_desc])

        # Reset the index to make 'date' a regular column
        df_RISMA.reset_index(inplace=True)

        # Keep rows with '0 to 5 cm depth'
        df_RISMA = df_RISMA[df_RISMA['depth'] == depth]

        # Interpolate mean_airt and mean_sst
        df_RISMA['mean_airt'] = df_RISMA['mean_airt'].interpolate()
        df_RISMA['mean_sst'] = df_RISMA['mean_sst'].interpolate()

        # Group the DataFrame by 'date' and 'station'
        grouped = df_RISMA.groupby(['date', 'station', 'depth', 'op']).apply(lambda x: x)
        
        grouped['category'] = grouped.apply(self.categorize_sensor, axis=1)

        # pivot the table
        pivoted = grouped.pivot(index=['date', 'station', 'depth', 'op', 'mean_sst', 'mean_airt'], columns='category', values='value')
        pivoted.reset_index(inplace=True)

        # add year
        pivoted['year'] = pivoted['date'].dt.year

        # add day of year as doy
        pivoted['doy'] = pivoted['date'].dt.dayofyear

        # Merge the dataframes based on the 'Station' column
        df_RISMA = pd.merge(pivoted, self.df_texture, left_on='station', right_on='Station', how='left')

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
        Load the Risma soil data into a DataFrame.
        """
        # Load the soil data from the predefined dictionary
        # This file contains the soil data for the Risma project, including soil properties at various depths for multiple stations.
        rima_stations = {
            "MB1": [
                {"depth_cm": 5, "density_gcm3": 1.28, "sand_pct": 78.8, "silt_pct": 10.1, "clay_pct": 11.1, "classification": "Sandy Loam"},
                {"depth_cm": 20, "density_gcm3": 1.56, "sand_pct": 80.2, "silt_pct": 8.3, "clay_pct": 11.5, "classification": "Sandy Loam"},
                {"depth_cm": 50, "density_gcm3": 1.5, "sand_pct": 81.5, "silt_pct": 8.2, "clay_pct": 10.3, "classification": "Loamy Sand"},
                {"depth_cm": 100, "density_gcm3": 1.57, "sand_pct": 81.0, "silt_pct": 6.8, "clay_pct": 12.2, "classification": "Sandy Loam"}
            ],
            "MB2": [
                {"depth_cm": 5, "density_gcm3": 1.35, "sand_pct": 44.9, "silt_pct": 20.8, "clay_pct": 34.3, "classification": "Clay Loam"},
                {"depth_cm": 20, "density_gcm3": 1.63, "sand_pct": 62.6, "silt_pct": 12.9, "clay_pct": 24.5, "classification": "Sandy Clay Loam"},
                {"depth_cm": 50, "density_gcm3": 1.63, "sand_pct": 65.8, "silt_pct": 13.2, "clay_pct": 21.0, "classification": "Sandy Clay Loam"},
                {"depth_cm": 100, "density_gcm3": 1.57, "sand_pct": 75.3, "silt_pct": 15.0, "clay_pct": 9.7, "classification": "Sandy Loam"}
            ],
            "MB3": [
                {"depth_cm": 5, "density_gcm3": 1.47, "sand_pct": 47.1, "silt_pct": 21.1, "clay_pct": 31.8, "classification": "Sandy Clay Loam"},
                {"depth_cm": 20, "density_gcm3": 1.52, "sand_pct": 45.8, "silt_pct": 21.3, "clay_pct": 32.9, "classification": "Sandy Clay Loam"},
                {"depth_cm": 50, "density_gcm3": 1.44, "sand_pct": 31.3, "silt_pct": 23.7, "clay_pct": 45.0, "classification": "Clay"},
                {"depth_cm": 100, "density_gcm3": 1.41, "sand_pct": 69.9, "silt_pct": 17.7, "clay_pct": 12.4, "classification": "Sandy Loam"}
            ],
            "MB4": [
                {"depth_cm": 5, "density_gcm3": 1.33, "sand_pct": 90.4, "silt_pct": 0.2, "clay_pct": 9.4, "classification": "Sand"},
                {"depth_cm": 20, "density_gcm3": 1.50, "sand_pct": 88.7, "silt_pct": 1.6, "clay_pct": 9.7, "classification": "Loamy Sand"},
                {"depth_cm": 50, "density_gcm3": 1.60, "sand_pct": 88.7, "silt_pct": 1.6, "clay_pct": 9.7, "classification": "Loamy Sand"},
                {"depth_cm": 100, "density_gcm3": 1.58, "sand_pct": 85.6, "silt_pct": 5.0, "clay_pct": 9.4, "classification": "Loamy Sand"}
            ],
            "MB5": [
                {"depth_cm": 5, "density_gcm3": 1.46, "sand_pct": 41.4, "silt_pct": 18.1, "clay_pct": 40.5, "classification": "Clay"},
                {"depth_cm": 20, "density_gcm3": 1.41, "sand_pct": 22.7, "silt_pct": 19.5, "clay_pct": 57.8, "classification": "Clay"},
                {"depth_cm": 50, "density_gcm3": 1.33, "sand_pct": 4.3, "silt_pct": 27.2, "clay_pct": 68.5, "classification": "Heavy Clay"},
                {"depth_cm": 100, "density_gcm3": 1.32, "sand_pct": 3.0, "silt_pct": 27.7, "clay_pct": 69.3, "classification": "Heavy Clay"}
            ],
            "MB6": [
                {"depth_cm": 5, "density_gcm3": 1.21, "sand_pct": 3.7, "silt_pct": 24.6, "clay_pct": 71.7, "classification": "Heavy Clay"},
                {"depth_cm": 20, "density_gcm3": 1.39, "sand_pct": 3.8, "silt_pct": 21.3, "clay_pct": 74.9, "classification": "Heavy Clay"},
                {"depth_cm": 50, "density_gcm3": 1.31, "sand_pct": 2.0, "silt_pct": 25.9, "clay_pct": 72.1, "classification": "Heavy Clay"},
                {"depth_cm": 100, "density_gcm3": 1.31, "sand_pct": 0.5, "silt_pct": 27.3, "clay_pct": 72.2, "classification": "Heavy Clay"}
            ],
            "MB7": [
                {"depth_cm": 5, "density_gcm3": 1.40, "sand_pct": 78.3, "silt_pct": 9.2, "clay_pct": 12.5, "classification": "Sandy Loam"},
                {"depth_cm": 20, "density_gcm3": 1.59, "sand_pct": 82.3, "silt_pct": 5.8, "clay_pct": 11.9, "classification": "Loamy Sand"},
                {"depth_cm": 50, "density_gcm3": 1.57, "sand_pct": 78.1, "silt_pct": 8.6, "clay_pct": 13.3, "classification": "Sandy Loam"},
                {"depth_cm": 100, "density_gcm3": 1.58, "sand_pct": 80.3, "silt_pct": 8.0, "clay_pct": 11.7, "classification": "Sandy Loam"}
            ],
            "MB8": [
                {"depth_cm": 5, "density_gcm3": 1.22, "sand_pct": 3.6, "silt_pct": 33.2, "clay_pct": 63.2, "classification": "Heavy Clay"},
                {"depth_cm": 20, "density_gcm3": 1.38, "sand_pct": 3.5, "silt_pct": 22.6, "clay_pct": 73.9, "classification": "Heavy Clay"},
                {"depth_cm": 50, "density_gcm3": 1.38, "sand_pct": 3.9, "silt_pct": 23.3, "clay_pct": 72.8, "classification": "Heavy Clay"},
                {"depth_cm": 100, "density_gcm3": 1.50, "sand_pct": 1.5, "silt_pct": 27.6, "clay_pct": 70.9, "classification": "Heavy Clay"}
            ],
            "MB9": [
                {"depth_cm": 5, "density_gcm3": 1.53, "sand_pct": 81.3, "silt_pct": 6.0, "clay_pct": 12.7, "classification": "Sandy Loam"},
                {"depth_cm": 20, "density_gcm3": 1.60, "sand_pct": 85.5, "silt_pct": 3.2, "clay_pct": 11.3, "classification": "Loamy Sand"},
                {"depth_cm": 50, "density_gcm3": 1.53, "sand_pct": 83.4, "silt_pct": 5.1, "clay_pct": 11.5, "classification": "Loamy Sand"},
                {"depth_cm": 100, "density_gcm3": 1.58, "sand_pct": 87.5, "silt_pct": 6.7, "clay_pct": 5.8, "classification": "Loamy Sand"}
            ],
            "MB10": [
                {"depth_cm": 5, "density_gcm3": 1.05, "sand_pct": 4.55, "silt_pct": 24.0, "clay_pct": 71.6, "classification": "Heavy Clay"},
                {"depth_cm": 20, "density_gcm3": 1.27, "sand_pct": 2.4, "silt_pct": 13.5, "clay_pct": 84.1, "classification": "Heavy Clay"},
                {"depth_cm": 50, "density_gcm3": 1.24, "sand_pct": 6.7, "silt_pct": 22.9, "clay_pct": 70.4, "classification": "Heavy Clay"},
                {"depth_cm": 100, "density_gcm3": 1.30, "sand_pct": 13.6, "silt_pct": 44.6, "clay_pct": 41.8, "classification": "Silty Clay"}
            ],
            "MB11": [
                {"depth_cm": 5, "density_gcm3": 1.3, "sand_pct": 23.8, "silt_pct": 39.5, "clay_pct": 36.8, "classification": "Clay Loam"},
                {"depth_cm": 20, "density_gcm3": 1.59, "sand_pct": 25.4, "silt_pct": 38.8, "clay_pct": 35.8, "classification": "Clay Loam"},
                {"depth_cm": 50, "density_gcm3": 1.64, "sand_pct": 24.0, "silt_pct": 46.6, "clay_pct": 29.3, "classification": "Clay Loam"},
                {"depth_cm": 100, "density_gcm3": 1.74, "sand_pct": 64.9, "silt_pct": 29.4, "clay_pct": 5.8, "classification": "Sandy Loam"}
            ],
            "MB12": [
                {"depth_cm": 5, "density_gcm3": 1.33, "sand_pct": 42.1, "silt_pct": 41.7, "clay_pct": 16.2, "classification": "Loam"},
                {"depth_cm": 20, "density_gcm3": 1.57, "sand_pct": 36.5, "silt_pct": 38.7, "clay_pct": 24.8, "classification": "Loam"},
                {"depth_cm": 50, "density_gcm3": 1.65, "sand_pct": 27.9, "silt_pct": 45.0, "clay_pct": 27.1, "classification": "Loam"},
                {"depth_cm": 100, "density_gcm3": 1.55, "sand_pct": 24.9, "silt_pct": 46.2, "clay_pct": 28.9, "classification": "Clay Loam"}
            ],
            "MB13": [
                {"depth_cm": 5, "density_gcm3": 1.3, "sand_pct": 81.0, "silt_pct": 12.0, "clay_pct": 7.0, "classification": "Loamy Sand"},
                {"depth_cm": 20, "density_gcm3": 1.6, "sand_pct": 81.0, "silt_pct": 11.0, "clay_pct": 8.0, "classification": "Loamy Sand"},
                {"depth_cm": 50, "density_gcm3": 1.52, "sand_pct": 73.0, "silt_pct": 15.0, "clay_pct": 12.0, "classification": "Sandy Loam"},
                {"depth_cm": 100, "density_gcm3": 1.32, "sand_pct": 19.0, "silt_pct": 39.0, "clay_pct": 42.0, "classification": "Clay"}
            ],
            "MB14": [
                {"depth_cm": 5, "density_gcm3": 1.4, "sand_pct": 89.0, "silt_pct": 7.0, "clay_pct": 4.0, "classification": "Sand"},
                {"depth_cm": 20, "density_gcm3": 1.4, "sand_pct": 73.0, "silt_pct": 17.0, "clay_pct": 10.0, "classification": "Sandy Loam"},
                {"depth_cm": 50, "density_gcm3": 1.4, "sand_pct": 49.0, "silt_pct": 31.0, "clay_pct": 20.0, "classification": "Loam"},
                {"depth_cm": 100, "density_gcm3": 1.4, "sand_pct": 23.0, "silt_pct": 43.0, "clay_pct": 34.0, "classification": "Clay Loam"}
            ],
            "MB15": [
                {"depth_cm": 5, "density_gcm3": 1.52, "sand_pct": 83.0, "silt_pct": 11.0, "clay_pct": 6.0, "classification": "Loamy Sand"},
                {"depth_cm": 20, "density_gcm3": 1.5, "sand_pct": 75.0, "silt_pct": 15.0, "clay_pct": 10.0, "classification": "Sandy Loam"},
                {"depth_cm": 50, "density_gcm3": 1.5, "sand_pct": 89.0, "silt_pct": 9.0, "clay_pct": 8.0, "classification": "Loamy Sand"},
                {"depth_cm": 100, "density_gcm3": 1.42, "sand_pct": 97.0, "silt_pct": 3.0, "clay_pct": 0.0, "classification": "Sand"}
            ]
        }

        # Create a list of dictionaries, each including the station name and measurement data
        all_data = [
            {"Station": station, **measurement}
            for station, measurements in rima_stations.items()
            for measurement in measurements
        ]

        # Convert the list to a DataFrame
        df_texture = pd.DataFrame(all_data)

        # Rename columns for better readability
        df_texture = df_texture.rename(columns={
            "depth_cm": "Depth (cm)",
            "density_gcm3": "bulk",
            "sand_pct": "Sand",
            "silt_pct": "Silt",
            "clay_pct": "Clay",
            "classification": "Classification"
        })

        # Keep rows with 5 cm depth
        df_texture = df_texture[df_texture['Depth (cm)'] == depth]

        # Normalize soil texture percentages to 0 to 1 range
        df_texture['Sand'] = df_texture['Sand'] / 100
        df_texture['Silt'] = df_texture['Silt'] / 100
        df_texture['Clay'] = df_texture['Clay'] / 100

        return df_texture


    def read_risma_bulk_csv(self, fname, station, S1_lot='18:30'):

        df = pd.read_csv(fname, header=None, low_memory=False)
        df = df.drop(index=[0,1,2,3,5])
        df = df.reset_index(drop=True)
        df[0] = pd.to_datetime(df[0], format='%Y-%m-%d %H:%M:%S')

        # Set the datetime column as the index
        df.set_index(df.columns[0], inplace=True)

        df.columns = df.iloc[0]
        df = df.iloc[1:]

        # remove first part of the columns name
        df.columns = [col.split('.')[1] for col in df.columns]
        # print(df.columns)

        # create a date object from 18:30
        overpass_time = datetime.strptime(S1_lot, '%H:%M')
        oh_bf_overpass_time = overpass_time - timedelta(hours=1)
        oh_af_overpass_time = overpass_time + timedelta(hours=1)

        # Convert the soil moisture columns to numeric, handling non-numeric values
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # 'coerce' will set non-numeric values to NaN

        # calculate mean daily soil temperature first group hourly rows into daily rows
        df_stats_air = df['Air temperature'].resample('D').agg(['min', 'max'])
        df_stats_air['mean_airt'] = df_stats_air['min'].add(df_stats_air['max']).div(2.)
        df_stats_air.drop(columns=['min', 'max'], inplace=True)

        if not 'Soil temperature 0 to 5 cm depth' in df.columns:
            sst_var = 'Soil temperature 5 cm depth'
        else:
            sst_var = 'Soil temperature 0 to 5 cm depth'

        df_stats_sst = df[sst_var].resample('D').agg(['min', 'max'])
        df_stats_sst['mean_sst'] = df_stats_sst['min'].add(df_stats_sst['max']).div(2.)
        df_stats_sst.drop(columns=['min', 'max'], inplace=True)

        # Keep rows around -1 and +1 overpass_time based on time index in the df
        df = df[(df.index.time >= oh_bf_overpass_time.time()) & (df.index.time <= oh_af_overpass_time.time())]

        # filter df's columns contain 'Soil water content' and 'Soil temperature'
        df = df.filter(regex='Soil water content|Soil temperature')

        # fill nan
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        # create a dict of columns as key and reducer 'median' as value
        aggregate = dict(zip(df.columns, ['mean', ] * len(df.columns)))

        # Resample the data by day, calculating mean for soil moisture and sum for precipitation
        daily_data = df.resample('D').agg(aggregate)

        # Melt the DataFrame to long format for plotting, including precipitation
        df_melted = daily_data.melt(ignore_index=False, var_name='sensor', value_name='value')
        df_melted.index.name = 'date'

        # reset index inplace and set doy as index
        df_melted.reset_index(inplace=True)
        df_melted.set_index('date', inplace=True)

        df_melted['station'] = station

        # add new column depth based on sensor
        df_melted['depth'] = df_melted['sensor'].apply(lambda x: re.findall(r"\d+ to \d+ cm|\d+ cm", x)[0])

        # merge df_melted and df_stats based on date
        df_melted = df_melted.merge(df_stats_air, left_index=True, right_index=True, how='left')
        df_melted = df_melted.merge(df_stats_sst, left_index=True, right_index=True, how='left')

        return df_melted


if __name__ == "__main__":
    # Example usage
    workspace_dir = './data'
    risma_dir = 'RISMA_CSV_SSM_SST_AirT_2015_2023'
    
    risma_data = RismaData(workspace_dir, risma_dir)
    df_risma = risma_data.load_df()
    
    print(df_risma.head())
    print("Risma data loaded successfully.")