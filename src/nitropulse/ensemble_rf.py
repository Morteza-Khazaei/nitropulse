from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import ee
import re
import csv
import os
import uuid
import geemap
import numpy as np
from geemap import ml
from tqdm.auto import tqdm
import pandas as pd


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
        self.workspace_dir = workspace_dir
        self.df = df.copy()
        self.df.columns = [self._normalize_column_name(c) for c in self.df.columns]
        # Drop duplicate columns that can arise from merges to keep single copies
        self.df = self.df.loc[:, ~pd.Index(self.df.columns).duplicated()]
        self._trained_feature_map = {}
        self._holdout_by_year = False
        self._test_year_map = None
    

    def run(self, vars, n_estimators=15, random_state=42, max_depth=15, holdout_by_year=True):
        trained_models = {}
        self._holdout_by_year = holdout_by_year
        if holdout_by_year:
            self._init_holdout_years(random_state)
        else:
            self._test_year_map = None
        for var in tqdm(vars, desc="Training ensemble models"):
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)
            trained_models[var] = self.__train(model, var)
            
        return trained_models
    
    def __features_for_var(self, var):
        var_lower = str(var).lower()
        if var_lower == 'ssm':
            return ['angle', 'vvs', 's', 'l', 'year', 'doy', 'lc', 'op']
        return ['vh', 'vv', 'angle', 'rvi', 'year', 'doy', 'lc', 'op']

    def __split_data(self, var, test_size=0.2, random_state=42):
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
        
        features = self.__features_for_var(var)
        self._trained_feature_map[var] = features

        # Print the features being used for training
        tqdm.write(f"Training model for '{var}' using features: {features}")

        var_lower = str(var).lower()
        if var_lower not in self.df.columns:
            available = ', '.join(sorted(self.df.columns))
            raise KeyError(f"Target column '{var}' (as '{var_lower}') not found. Available columns: {available}")

        X = self.df[features]
        y = self.df[var_lower]

        if self._holdout_by_year:
            if self._test_year_map is None:
                raise RuntimeError("Holdout years not initialised; call run() with holdout_by_year=True before training.")

            if 'station' not in self.df.columns or 'year' not in self.df.columns:
                raise KeyError("Columns 'station' and 'year' are required for year-based holdout splits.")

            test_years = self.df['station'].map(self._test_year_map)
            test_mask = test_years.eq(self.df['year']).fillna(False)

            X_train, X_test = X[~test_mask], X[test_mask]
            y_train, y_test = y[~test_mask], y[test_mask]

            if X_train.empty or X_test.empty:
                raise ValueError("Year-based holdout produced an empty train/test split. Check station/year coverage.")

            return X_train, X_test, y_train, y_test

        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def __train(self, model, var):
        """
        Trains the Random Forest Regressor model.

        Parameters:
        -----------
        X_train : array-like or sparse matrix of shape (n_samples, n_features)
            The training input samples.
        y_train : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers).
        """
        split = self.__split_data(var)
        X_train, X_test, y_train, y_test = split
        reg_rf = model.fit(X_train, y_train)

        # Make predictions on the holdout set
        y_pred = model.predict(X_test)

        # Evaluate the model
        r2 = model.score(X_test, y_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        bias= np.mean(y_test - y_pred)
        ubrmse = np.sqrt(max(rmse**2 - bias**2, 0))

        # Using tqdm.write to not interfere with progress bars
        tqdm.write(f"Metrics for {var}:")
        tqdm.write(f'\tR2: {r2:.4f}')
        tqdm.write(f'\tMAE: {mae:.4f}')
        tqdm.write(f'\tMSE: {mse:.4f}')
        tqdm.write(f'\tRMSE: {rmse:.4f}')
        tqdm.write(f'\tUBRMSE: {ubrmse:.4f}')
        tqdm.write(f'\tBias: {bias:.4f}')
        tqdm.write("")

        payload = {
            'model': reg_rf,
            'metrics': {
                'r2': r2,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'ubrmse': ubrmse,
                'bias': bias,
            }
        }

        if self._holdout_by_year:
            target_name = str(var).lower()
            base_test = self.df.loc[X_test.index].copy()
            base_test[f'{target_name}_pred'] = y_pred
            payload['test_df'] = base_test.reset_index(drop=True)

        return payload

    def __clean_tree(self, tree):
        return re.sub(r'\s+', ' ', tree)  # Replace multiple spaces with a single space

    def _init_holdout_years(self, random_state):
        if 'station' not in self.df.columns or 'year' not in self.df.columns:
            raise KeyError("Columns 'station' and 'year' are required for year-based holdout splits.")

        rng = np.random.default_rng(random_state)
        self._test_year_map = {}

        selections = []
        for station, group in self.df.groupby('station'):
            years = group['year'].dropna().unique()
            if len(years) == 0:
                continue
            chosen = rng.choice(years)
            self._test_year_map[station] = chosen
            selections.append(f"{station}:{chosen}")

        if not self._test_year_map:
            raise ValueError("No station/year combinations available to create holdout set.")

        summary = ', '.join(sorted(selections))
        tqdm.write(f"Holdout years (per station): {summary}")

    @staticmethod
    def _normalize_column_name(col):
        if isinstance(col, tuple):
            parts = [str(p) for p in col if p is not None and str(p) != '']
            col = '_'.join(parts)
        if col is None:
            return ''
        return str(col).lower()

    def __asset_exists(self, asset_id):
        try:
            ee.data.getAsset(asset_id)
            return True
        except Exception:
            return False

    def __resolve_backscatter_tag(self, rt_models):
        if not rt_models:
            return 'ENSEMBLE'
        tag = str(rt_models.get('RT_s', 'ENSEMBLE')).strip()
        sanitized = re.sub(r'[^A-Za-z0-9_]+', '', tag.upper())
        return sanitized if sanitized else 'ENSEMBLE'

    def __unique_asset_name(self, asset_root, base_name):
        while True:
            suffix = uuid.uuid4().hex[:6]
            candidate = f"{base_name}_{suffix}"
            asset_id = f"{asset_root}/{candidate}"
            if not self.__asset_exists(asset_id):
                return candidate, asset_id

    def upload_rf_to_gee(self, tranied_rfs, gee_project_id, rt_models=None, save_dectree=False):
        # Initialize the Earth Engine API.
        # This is done here to ensure it's initialized even if this module is run standalone.
        try:
            ee.Initialize(project=gee_project_id)
        except Exception:
            # If the above fails (e.g., in some environments), geemap provides a robust fallback.
            geemap.ee_initialize(project=gee_project_id)

        asset_root = f"projects/{gee_project_id}/assets"

        for var, model in tqdm(tranied_rfs.items(), desc="Uploading models to GEE"):
            n_estimators = model.n_estimators
            mdepth = model.max_depth
            backscatter_tag = self.__resolve_backscatter_tag(rt_models)
            base_name = f"{backscatter_tag}_ensemble_trees_{var}_n{n_estimators}_md{mdepth}"
            rfname, asset_id = self.__unique_asset_name(asset_root, base_name)
            feature_names = self._trained_feature_map.get(var, list(model.feature_names_in_))
            print(f"Uploading model for '{var}' as asset '{asset_id}' with features: {feature_names}")
            
            # Ensure all expected features are present in the model
            missing = set(feature_names) - set(model.feature_names_in_)
            if missing:
                raise ValueError(
                    f"Trained model for '{var}' is missing expected features: {sorted(missing)}."
                )

            # This is the critical function from geemap that performs the complex
            # serialization of the scikit-learn model into a GEE-compatible format.
            # Re-implementing this would be a significant effort.
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
                # Replaced geemap.ml.trees_to_csv with standard Python file I/O
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['tree'])
                    for tree_string in trees:
                        writer.writerow([tree_string])
        
        return None
