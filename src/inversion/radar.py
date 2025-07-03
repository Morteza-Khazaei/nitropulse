import os
import pandas as pd



class S1Data:
    """
    Class to handle S1 soil data processing.
    """
    def __init__(self, s1_dir):
        """
        Initialize the S1Data class.
        """
        self.s1_dir = s1_dir


    def load_df(self):
        """
        Load the S1 soil data from CSV files in the specified directory.
        This method processes both ascending and descending data from the S1 project.
        It reads CSV files, processes the data, and returns a DataFrame containing the processed data.
        """

        list_of_dfs = []

        # Loop through each file in the directory
        for filename in os.listdir(self.s1_dir):
            if filename.endswith(".csv"):  # Check if the file is a CSV file
                station_name = filename.split('_')[1]
                filepath = os.path.join(self.s1_dir, filename)
                print(f"Processing file: {filename}")
                df_S1 = self.read_S1_sigma_csv(filepath, station_name)

                list_of_dfs.append(df_S1)
        
        # Concatenate all DataFrames in the list into a single DataFrame
        df_S1_cat = pd.concat(list_of_dfs)

        # Reset the index to make 'date' a regular column
        df_S1_cat.reset_index(inplace=True)

        # rename landcover to lc
        df_S1_cat.rename(columns={'landcover': 'lc'}, inplace=True)

        df_S1_cat_gp = df_S1_cat.groupby(['year', 'doy', 'op', 'station'], group_keys=False).apply(lambda x: x).reset_index(drop=True)

        return df_S1_cat_gp


    def read_S1_sigma_csv(self, fname, station):
        
        
        df = pd.read_csv(fname)

        # drop .geo inplace
        df.drop(['system:index', '.geo'], axis=1, inplace=True)

        df_t = df.T
        df_t.index.rename('date', inplace=True)
        df_t.reset_index(inplace=True)

        # Add new column named 'band' by using of _** from20200902T001505_VH
        df_t['band'] = df_t['date'].apply(lambda x: x.split('_')[1])

        df_t['op'] = df_t['date'].apply(lambda x: x.split('_')[2])

        # if orbitpass == 'asc' : 0 else: 1
        df_t['op'] = df_t['op'].apply(lambda x: 0 if x == 'asc' else 1)

        # convert date to datetime by removing _** from20200902T001505_VH
        df_t['date'] = pd.to_datetime(df_t['date'].apply(lambda x: x.split('_')[0][:8]))

        # add year
        df_t['year'] = df_t['date'].dt.year

        # add day of year as doy
        df_t['doy'] = df_t['date'].dt.dayofyear

        # drop date
        df_t.drop('date', axis=1, inplace=True)

        # Delete all rows with column 'band' has value doy to year
        indices = df_t[(df_t['band'] == 'doy') | (df_t['band'] == 'year')].index
        df_t.drop(indices, inplace=True)

        # Groupby df_t based on orbitpass, year, and doy
        gp_df = df_t.groupby(['op', 'year', 'doy']).apply(lambda x: x.set_index('band').drop(['op', 'year', 'doy'], axis=1).T).reset_index()

        gp_df['station'] = station

        return gp_df