from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import ee
import re
import os
import geemap
import numpy as np
from geemap import ml
from tqdm.auto import tqdm


class RegressionRF:
    """
    A class for performing Random Forest Regression.
    """
    def __init__(self, workspace_dir, df):
        """
        Initializes the RegressionRF with specified parameters.

        Parameters:
        -----------
        n_estimators : int, optional
            The number of trees in the forest. The default is 100.
        random_state : int, optional
            Controls both the randomness of the bootstrapping of the samples used when building trees
            (if bootstrap=True) and the sampling of the features to consider when looking for the best split
            at each node (if max_features < n_features). The default is 42.
        """
        self.model = RandomForestRegressor(n_estimators=15, random_state=42, max_depth=15)
        self.workspace_dir = workspace_dir
        self.df = df
    

    def run(self, vars):
        trained_models = {}
        for var in tqdm(vars, desc="Training ensemble models"):
            trained_models[var] = self.__train(var)
            
        return trained_models
    
    def __split_data(self, df, var, test_size=0.2, random_state=42):
        """
        Splits the data into training and testing sets.

        Parameters:
        -----------
        df : pandas.DataFrame
            The input data.
        var : str
            The variable to predict.
        test_size : float, optional
            The proportion of the dataset to include in the test split. The default is 0.2.
        random_state : int, optional
            Controls the randomness of the data split. The default is 42.

        Returns:
        --------
        tuple
            A tuple containing the training and testing sets.
        """
        
        if var == 'ssm':
            features = ['angle', 'vvs', 's', 'l', 'year', 'doy', 'lc', 'op']
        else:
            features = ['VH', 'VV', 'angle', 'rvi', 'year', 'doy', 'lc', 'op']
        
        X = df[features]
        y = df[var]
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def __train(self, var):
        """
        Trains the Random Forest Regressor model.

        Parameters:
        -----------
        X_train : array-like or sparse matrix of shape (n_samples, n_features)
            The training input samples.
        y_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers).
        """
        X_train, X_test, y_train, y_test = self.__split_data(self.df, var)
        reg_rf = self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Evaluate the model
        r2 = self.model.score(X_test, y_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        bias= np.mean(y_test - y_pred)
        ubrmse = np.sqrt(rmse**2 - bias**2)

        # Using tqdm.write to not interfere with progress bars
        tqdm.write(f"Metrics for {var}:")
        tqdm.write(f'\tR2: {r2:.4f}')
        tqdm.write(f'\tMAE: {mae:.4f}')
        tqdm.write(f'\tMSE: {mse:.4f}')
        tqdm.write(f'\tRMSE: {rmse:.4f}')
        tqdm.write(f'\tUBRMSE: {ubrmse:.4f}')
        tqdm.write(f'\tBias: {bias:.4f}')
        tqdm.write("")

        return reg_rf
    
    def __clean_tree(self, tree):
        return re.sub(r'\s+', ' ', tree)  # Replace multiple spaces with a single space
    
    def upload_rf_to_gee(self, tranied_rfs, gee_project_id, save_dectree=False):
        if not ee.data._credentials:
            geemap.ee_initialize(project=gee_project_id)
            print("Earth Engine initialized.")
        else:
            print("Earth Engine already initialized.")
        
        user_id = geemap.ee_user_id()
        user_id = '/'.join(user_id.split('/')[:-1])

        for var, model in tqdm(tranied_rfs.items(), desc="Uploading models to GEE"):
            n_estimators = model.n_estimators
            mdepth = model.max_depth
            rfname = f"ensemble_trees_{var}_n{n_estimators}_md{mdepth}"
            feature_names = model.feature_names_in_

            asset_id = '/'.join([user_id, rfname])

            # convert the estimator into a list of strings
            # this function also works with the ensemble.ExtraTrees estimator
            trees = ml.rf_to_strings(model, feature_names, processes=32, output_mode='REGRESSION')

            cleaned_trees = [self.__clean_tree(tree) for tree in trees]

            # Convert tree strings to a FeatureCollection
            dummy_geometry = ee.Geometry.Point([0, 0])  # Placeholder geometry
            features = [ee.Feature(dummy_geometry, {"tree": tree.replace("\n", "#")}) for tree in cleaned_trees]
            fc = ee.FeatureCollection(features)

            # get export task and start
            task = ee.batch.Export.table.toAsset(collection=fc, description=rfname, assetId=asset_id)
            task.start()

            if save_dectree:
                file_path = os.path.join(self.workspace_dir, 'outputs', rfname + '.csv')
                ml.trees_to_csv(trees, file_path)
        
        return None
