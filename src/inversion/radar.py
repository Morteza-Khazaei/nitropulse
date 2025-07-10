import os
import ee
import time
import gdown
import pandas as pd

from inversion.gdrive import get_service, create_folder, make_folder_public, get_shareable_link



class S1Data:
    """
    Class to handle S1 soil data processing.
    """
    def __init__(self, workspace_dir, s1_dir='S1_CSV_files' ,auto_download=False):
        """
        Initialize the S1Data class.
        """
        inputs_dir = os.path.join(workspace_dir, 'inputs')
        os.makedirs(inputs_dir, exist_ok=True)

        for folder in os.listdir(inputs_dir):
            if os.path.isdir(os.path.join(inputs_dir, folder)) and 'S1_CSV' in folder:
                print(f"Found GEE folder: {folder}")
                s1_dir = os.path.join(inputs_dir, s1_dir)

        # Check if risma_dir exits
        if not os.path.exists(s1_dir):
            os.makedirs(s1_dir)
        else:
            print(f"S1_CSV folder already exists: {s1_dir}")
        self.s1_dir = s1_dir 
        self.workspace_dir = workspace_dir
        
        self.s1_google_drive_dir = s1_dir
        self.auto_download = auto_download

    
    def download_S1_data(self, stations=None, buffer_distance=15, start_date='2010-01-01', end_date='2024-01-01', gee_project_id=None, roi_asset_id=None):
        """
        Download Sentinel-1 data for specified stations and process it.
        This method processes Sentinel-1 data for each station in the specified list.
        It applies various image processing techniques, including edge masking, conversion to power, application of the Refined Lee filter, conversion to dB, and image enhancement.
        The processed images are then exported to Google Drive as CSV files.
        """

        if stations is None:
            stations = ['MB1', 'MB2', 'MB3', 'MB4', 'MB5', 'MB6', 'MB7', 'MB8', 'MB9', 'MB10', 'MB11', 'MB12', 'MB13', 'MB14', 'MB15',]  # Default station if none provided
        
        # Initialize Earth Engine
        ee.Authenticate()
        ee.Initialize(project=gee_project_id)
        print(ee.String('Hello from the Earth Engine servers!').getInfo())

        roi_asset_id = f'projects/{gee_project_id}/assets/{roi_asset_id}'
        roi = ee.FeatureCollection(roi_asset_id)

        # Loop through each station ID in the list
        for station_id in stations:

            fname = f'S1_Backscatter_RISMA_{station_id}_{start_date.split("-")[0]}_to_{end_date.split("-")[0]}_buffer{buffer_distance}m'
            file_path = os.path.join(self.s1_dir, f'{fname}.csv')

            # Check if file exists
            if os.path.exists(file_path):
                print(f"The file '{file_path}' exists. \n\tSkipping download process for {fname}.")
                continue  # Skip to the next station if the file already exists
            else:
                print(f"The file '{file_path}' does not exist. \n\tProceeding with download process for {fname}.")

                if self.auto_download:
                    print('Running on local machine detected!')
                    print(f'Automatic downloading from Google Drive to {self.s1_dir}.')

                    # Authenticate and get the service
                    print('Authenticating with Google Drive...')
                    service = get_service(path=os.path.join(self.workspace_dir, 'config', 'gdrive'))

                    # Create the folder in Google Drive if it doesn't exist
                    folder_id = create_folder(service, self.s1_google_drive_dir)
                    make_folder_public(service, folder_id)

                # Define the region of interest (ROI) using the station ID and buffer distance
                geom_buffer = roi.filter(ee.Filter.eq('Station ID', station_id)).geometry().buffer(buffer_distance)

                # Combined function to add date and crop type efficiently
                def enhance_image(image):
                    date = image.date()
                    year = date.get('year')
                    year_image = ee.Image.constant(year).rename('year').toDouble()
                    doy_image = ee.Image.constant(date.getRelative('day', 'year').add(1)).rename('doy').toDouble()
                    crop_type = ee.ImageCollection('AAFC/ACI').filter(ee.Filter.calendarRange(year, year, 'year')).first().clip(geom_buffer)
                    return image.addBands([year_image, doy_image, crop_type])

                # Process Sentinel-1 collection with chaining
                S1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
                    .filterBounds(geom_buffer) \
                    .filterDate(ee.Date(start_date), ee.Date(end_date)) \
                    .filter(ee.Filter.And(
                        ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'),
                        ee.Filter.eq('instrumentMode', 'IW')
                    )) \
                    .map(self.apply_edge_mask) \
                    .map(self.to_power) \
                    .map(self.apply_refined_lee) \
                    .map(self.to_db) \
                    .map(enhance_image)

                S1 = ee.ImageCollection(S1.distinct('system:time_start'))

                desc = S1.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
                asc = S1.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
                print('Descending image size:', desc.size().getInfo())
                print('Ascending image size:', asc.size().getInfo())

                # Rename bands efficiently
                images_desc = self.rename_bands(desc, 'desc')
                images_asc = self.rename_bands(asc, 'asc')
                images = ee.Image.cat([images_desc, images_asc]).toShort()

                # Reduce regions instead of sample
                point_with_values = images.reduceRegions(
                    collection=geom_buffer,
                    reducer=ee.Reducer.median(),
                    scale=10,  # meters
                    tileScale=16
                )

                # Export to Google Drive as CSV
                task = ee.batch.Export.table.toDrive(
                    collection=point_with_values,
                    description=fname,
                    folder=self.s1_google_drive_dir,
                    fileFormat='CSV'
                )

                # Start the export task
                task.start()
                print(f'RISMA Stations {station_id}: Export task started with ID: {task.id}')

                status = task.status()
                state = status['state'].lower()
                previous_state = None  # Variable to store the previous state
                while state != 'completed':
                    
                    if state != previous_state:
                        print(f'\tProgress: Task is {state} on the GEE server.')
                        previous_state = state
                    
                    time.sleep(15) # Sleep for 15 seconds
                    status = task.status()
                    state = status['state'].lower()
                    
                print(f'\tProgress: Task is {state}.')
                fid = status['destination_uris'][0].split('/')[-1]

            if not self.in_colab_shell():
                # If running in Google Colab, download the folder from Google Drive
                # Download from Google Drive
                if self.auto_download:
                    # Download the folder using gdown
                    gdown.download_folder(id=fid, quiet=False, use_cookies=False, output=self.s1_dir)
                
                else:
                    print('Running on local machine detected!')
                    print(f'Please download manually the {self.s1_google_drive_dir} folder from Google Drive to {self.s1_dir}.')
                    
            else:
                print(f'Running on Google Colab detected. no need to download from Google Drive. The files are already in the folder {self.s1_google_drive_dir}.')


    def in_colab_shell(self,):
        """Tests if the code is being executed within Google Colab."""
        import sys

        if "google.colab" in sys.modules:
            return True
        else:
            return False
    
    def load_df(self):
        """
        Load the S1 soil data from CSV files in the specified directory.
        This method processes both ascending and descending data from the S1 project.
        It reads CSV files, processes the data, and returns a DataFrame containing the processed data.
        """

        list_of_dfs = []

        # Loop through each file in the directory
        for filename in sorted(os.listdir(self.s1_dir), key=lambda x:int(x.split('_')[1][2:])):
            if filename.endswith(".csv"):  # Check if the file is a CSV file
                station_name = filename.split('_')[1]
                filepath = os.path.join(self.s1_dir, filename)
                print(f"\tProcessing file: {filename}")
                df_S1 = self.read_S1_sigma_csv(filepath, station_name)

                list_of_dfs.append(df_S1)
        
        # Concatenate all DataFrames in the list into a single DataFrame
        df_S1_cat = pd.concat(list_of_dfs)

        # Reset the index to make 'date' a regular column
        df_S1_cat.reset_index(inplace=True)

        # rename landcover to lc
        df_S1_cat.rename(columns={'landcover': 'lc'}, inplace=True)

        # Cast lc to int then str
        df_S1_cat['lc'] = df_S1_cat['lc'].astype(int).astype(str)

        return df_S1_cat


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

        gp_df = df_t.pivot_table(
            index=['op', 'year', 'doy'],
            columns='band',
            values=df_t.columns.difference(['op', 'year', 'doy', 'band'])[0]  # usually the value column
        ).reset_index()
        
        gp_df['station'] = station

        return gp_df
    
    def rename_bands(self, image_collection, suffix):
        image = image_collection.toBands()
        
        def rename_band(name):
            parts = ee.String(name).split('_')
            return ee.String(parts.get(4)).cat('_').cat(parts.get(-1)).cat('_').cat(suffix)
        
        band_names = image.bandNames().map(rename_band)
        return image.rename(band_names)

    # Refined Lee speckle filter (same as in SNAP s1tbx!)
    def refined_lee(self, img):
        """
        img must be in natural units, i.e. not in dB!
        """
        # Set up 3x3 kernels 
        weights3 = ee.List.repeat(ee.List.repeat(1, 3), 3)
        kernel3 = ee.Kernel.fixed(3, 3, weights3, 1, 1, False)

        mean3 = img.reduceNeighborhood(ee.Reducer.mean(), kernel3)
        variance3 = img.reduceNeighborhood(ee.Reducer.variance(), kernel3)

        # Use a sample of the 3x3 windows inside a 7x7 windows to determine gradients and directions
        sample_weights = ee.List([
            [0,0,0,0,0,0,0], 
            [0,1,0,1,0,1,0],
            [0,0,0,0,0,0,0], 
            [0,1,0,1,0,1,0], 
            [0,0,0,0,0,0,0], 
            [0,1,0,1,0,1,0],
            [0,0,0,0,0,0,0]
        ])

        sample_kernel = ee.Kernel.fixed(7, 7, sample_weights, 3, 3, False)

        # Calculate mean and variance for the sampled windows and store as 9 bands
        sample_mean = mean3.neighborhoodToBands(sample_kernel)
        sample_var = variance3.neighborhoodToBands(sample_kernel)

        # Determine the 4 gradients for the sampled windows
        gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs()
        gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs())
        gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs())
        gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs())

        # And find the maximum gradient amongst gradient bands
        max_gradient = gradients.reduce(ee.Reducer.max())

        # Create a mask for band pixels that are the maximum gradient
        gradmask = gradients.eq(max_gradient)

        # duplicate gradmask bands: each gradient represents 2 directions
        gradmask = gradmask.addBands(gradmask)

        # Determine the 8 directions
        directions = sample_mean.select(1).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(7))).multiply(1)
        directions = directions.addBands(sample_mean.select(6).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2))
        directions = directions.addBands(sample_mean.select(3).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3))
        directions = directions.addBands(sample_mean.select(0).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4))
        # The next 4 are the not() of the previous 4
        directions = directions.addBands(directions.select(0).Not().multiply(5))
        directions = directions.addBands(directions.select(1).Not().multiply(6))
        directions = directions.addBands(directions.select(2).Not().multiply(7))
        directions = directions.addBands(directions.select(3).Not().multiply(8))

        # Mask all values that are not 1-8
        directions = directions.updateMask(gradmask)

        # "collapse" the stack into a single band image (due to masking, each pixel has just one value (1-8) in it's directional band, and is otherwise masked)
        directions = directions.reduce(ee.Reducer.sum())

        sample_stats = sample_var.divide(sample_mean.multiply(sample_mean))

        # Calculate localNoiseVariance
        sigma_v = sample_stats.toArray().arraySort().arraySlice(0, 0, 5).arrayReduce(ee.Reducer.mean(), [0])

        # Set up the 7*7 kernels for directional statistics
        rect_weights = ee.List.repeat(ee.List.repeat(0, 7), 3).cat(ee.List.repeat(ee.List.repeat(1, 7), 4))

        diag_weights = ee.List([
            [1,0,0,0,0,0,0], 
            [1,1,0,0,0,0,0], 
            [1,1,1,0,0,0,0], 
            [1,1,1,1,0,0,0], 
            [1,1,1,1,1,0,0], 
            [1,1,1,1,1,1,0], 
            [1,1,1,1,1,1,1]
        ])

        rect_kernel = ee.Kernel.fixed(7, 7, rect_weights, 3, 3, False)
        diag_kernel = ee.Kernel.fixed(7, 7, diag_weights, 3, 3, False)

        # Create stacks for mean and variance using the original kernels. Mask with relevant direction.
        dir_mean = img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1))
        dir_var = img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1))

        dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)))
        dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)))

        # and add the bands for rotated kernels
        for i in range(1, 4):
            dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
            dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
            dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))
            dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))

        # "collapse" the stack into a single band image (due to masking, each pixel has just one value in it's directional band, and is otherwise masked)
        dir_mean = dir_mean.reduce(ee.Reducer.sum())
        dir_var = dir_var.reduce(ee.Reducer.sum())

        # A finally generate the filtered value
        var_x = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigma_v)).divide(sigma_v.add(1.0))

        b = var_x.divide(dir_var)

        result = dir_mean.add(b.multiply(img.subtract(dir_mean)))
        return result.arrayFlatten([['sum']])

    # Function to convert from dB
    def to_power(self, image):
        bands = ee.Image(10.0).pow(image.select('V.*').divide(10.0))
        return image.addBands(bands, None, True)

    # Function to convert to dB
    def to_db(self, image):
        bands = image.select('V.*').log10().multiply(10.0)
        return image.addBands(bands, None, True)

    # Function to apply edge mask
    def apply_edge_mask(self, image):
        edge = image.lt(-30.0)
        return image.updateMask(image.mask().And(edge.Not()))

    # Function to apply Refined Lee filter
    def apply_refined_lee(self, image):
        proj = image.select('VV').projection()
        band_names = image.bandNames().remove('angle')
        
        def process_band(b):
            resampled = image.select([b]).resample('bicubic')
            return self.refined_lee(resampled)
        
        clean_image = ee.ImageCollection(band_names.map(process_band)).toBands().reproject(proj).rename(band_names).copyProperties(image)
        return image.addBands(clean_image, None, True)



if __name__ == "__main__":
    # Example usage
    s1 = S1Data(workspace_dir='./assets', auto_download=True)
    # s1.download_S1_data(stations=['MB1', ], buffer_distance=20, start_date='2010-01-01', end_date='2024-01-01', gee_project_id='ee-mortezakhazaei1370', roi_asset_id='RISMA_Stations_Canada')
    
    df = s1.load_df()
    print(df.head())