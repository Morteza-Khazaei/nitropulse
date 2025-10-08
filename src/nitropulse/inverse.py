import os
import json
import warnings
import hashlib
import re
import shutil
from datetime import datetime
from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.integrate import IntegrationWarning

from aiem import AIEM0
from ssrt import S2RTR
from ssrt.surface import PRISM1, SMART, SPM3D, I2EM_Bistat_model
from ssrt.utils import Dobson85




class Inverse:
    """
    Class to handle inversion data processing.
    """

    def __init__(self, workspace_dir, config, show_progress=False, progress_fn=None, suppress_warnings=True):
        """
        Initialize the Inverse class with directories, model configurations, and crop GDD thresholds.
        """
        self.workspace_dir = workspace_dir
        self.show_progress = show_progress
        self._progress_fn = progress_fn
        self.suppress_warnings = suppress_warnings
        
        self.config = config or {}
        if not isinstance(self.config, dict):
            raise TypeError("config must be provided as a dictionary")

        # Ensure config files exist in workspace
        self._ensure_config_files()

        try:
            self.fGHz = float(self.config['fGHz'])
        except KeyError as exc:
            raise KeyError("config must include 'fGHz'") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError("config['fGHz'] must be a numeric value") from exc

        models = self.config.get('models') or {}
        if not isinstance(models, dict):
            raise TypeError("config['models'] must be a dictionary")
        self.models = models

        self.acftype = str(self.config.get('acftype', 'exp'))

        iterations_value = self.config.get('iterations', 1)
        try:
            self.iterations = int(iterations_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("config['iterations'] must be an integer") from exc
        if self.iterations < 1:
            raise ValueError("config['iterations'] must be greater than or equal to 1")

        # Expose checkpoint tracking details after runs
        self.last_checkpoint_path = None
        self.last_bounds_snapshot_path = None
        self.last_checkpoint_manifest_path = None
        self.last_checkpoint_manifest = None
        self.last_completed_iteration = 0

        # Load it back as a dict
        crop_bbch_k_b_coff_file = os.path.join(self.workspace_dir, 'config', 'gdd', 'crop_bbch_k_b_coff.json')
        with open(crop_bbch_k_b_coff_file, "r") as f:
            crop_bbch_k_b_coff = json.load(f)
        self.crop_bbch_k_b_coff = crop_bbch_k_b_coff
        
        # Load crop bounds from the JSON file
        crop_bounds_file = os.path.join(self.workspace_dir, 'config', 'inversion', 'crop_inversion_bounds.json')
        with open(crop_bounds_file, "r") as f:
            crop_inversion_bounds = json.load(f)

        for crop_code, params in crop_inversion_bounds.items():
            for key in ("cbound", "sbound", "lbound", "wbound"):
                params[key] = self._validate_bound_triplet(crop_code, key, params[key])

        self.crop_inversion_bounds = crop_inversion_bounds

    def _ensure_config_files(self):
        """
        Ensure that required config files exist in the workspace directory.
        If they don't exist, copy them from the package config directory.
        """
        # Get the package config directory
        package_dir = os.path.dirname(os.path.abspath(__file__))
        package_config_dir = os.path.join(package_dir, 'config')
        
        # Define required config files
        required_files = [
            ('gdd', 'crop_bbch_k_b_coff.json'),
            ('gdd', 'crop_base_temp.json'),
            ('gdd', 'crop_gdd_thresh.json'),
            ('inversion', 'crop_inversion_bounds.json'),
            ('risma', 'stations_texture.json'),
        ]
        
        for subdir, filename in required_files:
            # Source file in package
            src_file = os.path.join(package_config_dir, subdir, filename)
            
            # Destination file in workspace
            dest_dir = os.path.join(self.workspace_dir, 'config', subdir)
            dest_file = os.path.join(dest_dir, filename)
            
            # Check if destination file exists
            if not os.path.exists(dest_file):
                # Create destination directory if it doesn't exist
                os.makedirs(dest_dir, exist_ok=True)
                
                # Copy file from package to workspace
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dest_file)
                    if self.show_progress:
                        print(f"Copied config file: {filename} to {dest_dir}")
                else:
                    warnings.warn(f"Source config file not found: {src_file}", UserWarning)


    def _run_single_iteration(self, pheno_df, bounds=None):
        total_records = len(pheno_df)
        grouped = pheno_df.groupby(['op', 'year', 'doy', 'angle'], group_keys=False)
        total_groups = grouped.ngroups

        self._emit_progress(f"Starting inversion on {total_records} records across {total_groups} groups")

        processed_groups = []
        active_bounds = bounds if bounds is not None else self.crop_inversion_bounds
        for idx, (_, group) in enumerate(grouped, start=1):
            processed_groups.append(self.inversion(group.copy(), active_bounds))
            self._emit_progress(f"Processed group {idx}/{total_groups}")

        if processed_groups:
            pheno_df = pd.concat(processed_groups)
        else:
            pheno_df = pheno_df.iloc[0:0]

        self._emit_progress("Inversion complete")

        # Keep rows where we produced a soil backscatter estimate (vvs)
        pheno_df = pheno_df[pheno_df['vvs'].notna()]

        # drop rows with vvs lower than -50
        pheno_df = pheno_df[pheno_df.vvs > -50]

        return pheno_df

    def run(self, pheno_df, iterations=None, output_dir=None, resume=True):
        if iterations is None:
            iterations = self.iterations
        try:
            iterations = int(iterations)
        except (TypeError, ValueError) as exc:
            raise ValueError("iterations must be an integer greater than zero") from exc
        if iterations < 1:
            raise ValueError("iterations must be an integer greater than zero")

        checkpoint_dir = self._get_checkpoint_dir(ensure=True)
        bounds_output_dir = output_dir or checkpoint_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        current_bounds = deepcopy(self.crop_inversion_bounds)
        latest_result = None
        start_iteration = 1
        last_checkpoint_path = None
        last_bounds_snapshot_path = None
        completed_iteration = 0

        if resume:
            restored_state = self._restore_iteration_state(iterations, output_dir=bounds_output_dir)
            if restored_state is not None:
                last_iter, last_result, restored_bounds, checkpoint_path, bounds_path = restored_state
                current_bounds = restored_bounds
                latest_result = last_result
                start_iteration = last_iter + 1
                completed_iteration = last_iter
                last_checkpoint_path = checkpoint_path
                last_bounds_snapshot_path = bounds_path
                self._emit_progress(f"Resuming from iteration {last_iter} checkpoint")

                if start_iteration > iterations:
                    self.crop_inversion_bounds = current_bounds
                    self.last_checkpoint_path = last_checkpoint_path
                    self.last_bounds_snapshot_path = last_bounds_snapshot_path
                    self.last_completed_iteration = completed_iteration
                    self.last_checkpoint_manifest_path = self._record_latest_checkpoint(
                        last_checkpoint_path,
                        last_bounds_snapshot_path,
                        completed_iteration,
                        iterations,
                        checkpoint_dir,
                    )
                    self._emit_progress("Checkpoint already satisfies requested iterations")
                    return latest_result

        for iteration in range(start_iteration, iterations + 1):
            self._emit_progress(f"Iteration {iteration}/{iterations} starting")
            latest_result = self._run_single_iteration(pheno_df.copy(), bounds=current_bounds)

            current_bounds = self._derive_bounds_from_results(latest_result, current_bounds)
            snapshot_path = self._write_bounds_snapshot(current_bounds, iteration, bounds_output_dir)
            checkpoint_path = self._write_iteration_checkpoint(latest_result, iteration)
            last_checkpoint_path = checkpoint_path
            last_bounds_snapshot_path = snapshot_path
            completed_iteration = iteration
            self._emit_progress(
                f"Iteration {iteration} bounds written to {snapshot_path} and checkpointed at {checkpoint_path}"
            )

        self.crop_inversion_bounds = current_bounds
        self.last_checkpoint_path = last_checkpoint_path
        self.last_bounds_snapshot_path = last_bounds_snapshot_path
        self.last_completed_iteration = completed_iteration
        self.last_checkpoint_manifest_path = self._record_latest_checkpoint(
            last_checkpoint_path,
            last_bounds_snapshot_path,
            completed_iteration,
            iterations,
            checkpoint_dir,
        )
        self._emit_progress("Iterative inversion complete")

        return latest_result


    def run_single(self, pheno_df, bounds=None):
        """Execute a single inversion iteration without checkpoint management."""
        return self._run_single_iteration(pheno_df, bounds=bounds)


    def to_power(self, dB):
        return 10**(dB/10)

    def to_dB(self, power):
        return 10*np.log10(power)

    @staticmethod
    def _validate_bound_triplet(crop_code, bound_name, triplet):
        if len(triplet) != 3:
            raise ValueError(f"{crop_code}:{bound_name} must contain three elements [lower, seed, upper]")

        lo, seed, hi = map(float, triplet)

        if not np.isfinite([lo, seed, hi]).all():
            raise ValueError(f"{crop_code}:{bound_name} contains non-finite values: {triplet}")

        if lo > hi:
            warnings.warn(
                f"{crop_code}:{bound_name} lower bound {lo} is greater than upper {hi}; swapping to maintain order.",
                UserWarning,
            )
            lo, hi = hi, lo

        if seed < lo:
            warnings.warn(
                f"{crop_code}:{bound_name} seed {seed} below lower bound {lo}; clamping to lower.",
                UserWarning,
            )
            seed = lo
        elif seed > hi:
            warnings.warn(
                f"{crop_code}:{bound_name} seed {seed} above upper bound {hi}; clamping to upper.",
                UserWarning,
            )
            seed = hi

        if lo == hi:
            hi = lo + 1e-6
            warnings.warn(
                f"{crop_code}:{bound_name} lower and upper bounds were equal; nudging upper to {hi}.",
                UserWarning,
            )

        return [lo, seed, hi]

    def _sanitize_token(self, value):
        token = str(value).strip().lower()
        token = re.sub(r'[^a-z0-9]+', '-', token)
        token = token.strip('-')
        return token or 'unknown'

    def _backscatter_model_tokens(self):
        tokens = []
        for key, value in sorted((self.models or {}).items()):
            if not str(key).lower().startswith('rt'):
                continue
            key_token = self._sanitize_token(key).replace('rt-', 'rt')
            tokens.append((key_token, self._sanitize_token(value)))
        if not tokens:
            tokens.append(('rt', 'unknown'))
        return tokens

    def _backscatter_suffix(self):
        parts = [f"{key}-{value}" for key, value in self._backscatter_model_tokens()]
        return '_'.join(parts)

    def _checkpoint_identifier(self):
        models_repr = json.dumps(self.models, sort_keys=True, default=str)
        digest = hashlib.sha256(models_repr.encode()).hexdigest()[:12]
        freq_label = str(self.fGHz).replace('.', 'p').replace('-', 'm')
        acf_label = self._sanitize_token(self.acftype)
        return f"f{freq_label}_ac_{acf_label}_{self._backscatter_suffix()}_{digest}"

    def _get_checkpoint_dir(self, ensure=False):
        base_dir = os.path.join(self.workspace_dir, 'inversion_checkpoints', self._checkpoint_identifier())
        if ensure:
            os.makedirs(base_dir, exist_ok=True)
            self._write_checkpoint_metadata(base_dir)
        return base_dir

    def _write_checkpoint_metadata(self, checkpoint_dir):
        metadata_path = os.path.join(checkpoint_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            return
        metadata = {
            'fGHz': self.fGHz,
            'acftype': self.acftype,
            'models_repr': json.dumps(self.models, sort_keys=True, default=str),
            'iterations': self.iterations,
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _checkpoint_filename(self, iteration_index):
        suffix = self._backscatter_suffix()
        return f"inversion_iter_{iteration_index:03d}_{suffix}.pkl"

    def _checkpoint_path(self, iteration_index, ensure_dir=False):
        directory = self._get_checkpoint_dir(ensure=ensure_dir)
        filename = self._checkpoint_filename(iteration_index)
        return os.path.join(directory, filename)

    def _list_checkpoint_iterations(self):
        directory = self._get_checkpoint_dir(ensure=False)
        if not os.path.isdir(directory):
            return []
        iterations = []
        pattern_new = re.compile(r'^inversion_iter_(\d+)_.*\.pkl$')
        pattern_old = re.compile(r'^inversion_iter_(\d+)\.pkl$')
        for entry in os.listdir(directory):
            match = pattern_new.match(entry) or pattern_old.match(entry)
            if not match:
                continue
            try:
                iterations.append(int(match.group(1)))
            except (TypeError, ValueError):
                continue
        return sorted(iterations)

    def _write_iteration_checkpoint(self, df, iteration_index):
        path = self._checkpoint_path(iteration_index, ensure_dir=True)
        df.to_pickle(path)
        return path

    def _load_iteration_checkpoint(self, iteration_index):
        path = self._resolve_checkpoint_path(iteration_index)
        if path is None:
            return None, None
        return pd.read_pickle(path), path

    def _resolve_checkpoint_path(self, iteration_index):
        directory = self._get_checkpoint_dir(ensure=False)
        if not os.path.isdir(directory):
            return None

        candidates = [
            os.path.join(directory, self._checkpoint_filename(iteration_index)),
            os.path.join(directory, f"inversion_iter_{iteration_index:03d}.pkl"),
            os.path.join(directory, f"inversion_iter_{iteration_index}.pkl"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        pattern_new = re.compile(rf'^inversion_iter_{iteration_index:03d}_.*\.pkl$')
        pattern_old = re.compile(rf'^inversion_iter_{iteration_index}\.pkl$')
        for entry in os.listdir(directory):
            if pattern_new.match(entry) or pattern_old.match(entry):
                return os.path.join(directory, entry)
        return None

    def _load_bounds_snapshot(self, iteration_index, output_dir=None):
        candidate_dirs = []
        if output_dir is not None:
            candidate_dirs.append(output_dir)
        # Fall back to the legacy location if needed
        candidate_dirs.append(os.path.join(self.workspace_dir, 'config', 'inversion'))
        for directory in candidate_dirs:
            if not directory:
                continue
            path = self._resolve_bounds_snapshot_path(directory, iteration_index)
            if path is None:
                continue
            with open(path, 'r') as f:
                data = json.load(f)
            return data, path
        return None, None

    def _bounds_snapshot_filename(self, iteration_index):
        suffix = self._backscatter_suffix()
        return f"crop_inversion_bounds_iter_{iteration_index:03d}_{suffix}.json"

    def _resolve_bounds_snapshot_path(self, directory, iteration_index):
        candidates = [
            os.path.join(directory, self._bounds_snapshot_filename(iteration_index)),
            os.path.join(directory, f"crop_inversion_bounds_iter_{iteration_index:03d}.json"),
            os.path.join(directory, f"crop_inversion_bounds_iter_{iteration_index}.json"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        pattern_new = re.compile(rf'^crop_inversion_bounds_iter_{iteration_index:03d}_.*\.json$')
        pattern_old = re.compile(rf'^crop_inversion_bounds_iter_{iteration_index}\.json$')
        for entry in os.listdir(directory):
            if pattern_new.match(entry) or pattern_old.match(entry):
                return os.path.join(directory, entry)
        return None

    def _normalize_bounds(self, bounds, base=None):
        normalized = deepcopy(base) if base is not None else {}
        for crop_code, crop_bounds in (bounds or {}).items():
            normalized.setdefault(crop_code, {})
            for bound_name in ('cbound', 'sbound', 'lbound', 'wbound'):
                triplet = crop_bounds.get(bound_name)
                if triplet is None:
                    continue
                normalized[crop_code][bound_name] = self._validate_bound_triplet(crop_code, bound_name, triplet)
        return normalized

    def _restore_iteration_state(self, iterations, output_dir=None):
        existing_iterations = [idx for idx in self._list_checkpoint_iterations() if idx <= iterations]
        if not existing_iterations:
            return None

        latest_iteration = max(existing_iterations)
        latest_result, checkpoint_path = self._load_iteration_checkpoint(latest_iteration)
        if latest_result is None:
            return None

        stored_bounds, bounds_path = self._load_bounds_snapshot(latest_iteration, output_dir=output_dir)
        if stored_bounds is not None:
            current_bounds = self._normalize_bounds(stored_bounds, base=deepcopy(self.crop_inversion_bounds))
        else:
            current_bounds = self._derive_bounds_from_results(latest_result, deepcopy(self.crop_inversion_bounds))
            bounds_path = None

        return latest_iteration, latest_result, current_bounds, checkpoint_path, bounds_path

    def _record_latest_checkpoint(self, checkpoint_path, bounds_path, iteration_index, total_iterations, checkpoint_dir):
        if checkpoint_path is None and bounds_path is None:
            return None

        safe_models = json.loads(json.dumps(self.models, default=str))
        manifest = {
            'checkpoint_path': checkpoint_path,
            'bounds_snapshot_path': bounds_path,
            'iteration': iteration_index,
            'total_iterations_requested': total_iterations,
            'identifier': self._checkpoint_identifier(),
            'checkpoint_dir': checkpoint_dir,
            'fGHz': self.fGHz,
            'acftype': self.acftype,
            'iterations_configured': self.iterations,
            'models': safe_models,
            'timestamp_utc': datetime.utcnow().replace(microsecond=0).isoformat() + 'Z',
        }

        manifest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.json')
        manifest['manifest_path'] = manifest_path

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        pointer_paths = [
            os.path.join(self.workspace_dir, 'inversion_checkpoints', 'latest.json'),
            os.path.join(self.workspace_dir, 'outputs', 'last_inversion_checkpoint.json'),
        ]
        for pointer_path in pointer_paths:
            os.makedirs(os.path.dirname(pointer_path), exist_ok=True)
            with open(pointer_path, 'w') as f:
                json.dump(manifest, f, indent=2)

        self.last_checkpoint_manifest = manifest
        return manifest_path

    def _emit_progress(self, message):
        if self._progress_fn is not None:
            self._progress_fn(message)
        elif self.show_progress:
            print(message)

    @contextmanager
    def _maybe_suppress_integration_warning(self):
        if self.suppress_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", IntegrationWarning)
                yield
        else:
            yield

    def _derive_bounds_from_results(self, pheno_df, previous_bounds):
        updated_bounds = deepcopy(previous_bounds) if previous_bounds is not None else {}
        if 'lc' not in pheno_df.columns:
            return updated_bounds

        param_to_bound = {
            'c': 'cbound',
            's': 'sbound',
            'l': 'lbound',
            'w': 'wbound',
        }

        grouped = pheno_df.groupby(pheno_df['lc'].astype(int).astype(str))

        for crop_code, group in grouped:
            crop_bounds = updated_bounds.setdefault(crop_code, {})
            for param, bound_name in param_to_bound.items():
                if param not in group:
                    continue
                values = pd.to_numeric(group[param], errors='coerce').dropna()
                if values.empty:
                    continue
                lower = float(values.min())
                seed = float(values.mean())
                upper = float(values.max())
                crop_bounds[bound_name] = [lower, seed, upper]

        for crop_code, crop_bounds in updated_bounds.items():
            for bound_name in ('cbound', 'sbound', 'lbound', 'wbound'):
                triplet = crop_bounds.get(bound_name)
                if triplet is None:
                    continue
                crop_bounds[bound_name] = self._validate_bound_triplet(crop_code, bound_name, triplet)

        return updated_bounds

    def _write_bounds_snapshot(self, bounds, iteration_index, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(self.workspace_dir, 'config', 'inversion')
        os.makedirs(output_dir, exist_ok=True)

        filename = self._bounds_snapshot_filename(iteration_index)
        path = os.path.join(output_dir, filename)

        with open(path, 'w') as f:
            json.dump(bounds, f, indent=4)

        return path

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
        if ph < 0:
            ph = 0

        # Define min and max for PH
        min_ph = ph - 0.001
        if min_ph < 0:
            min_ph = 0
        max_ph = ph + 0.001

        return min_ph, ph, max_ph, k, b
    
    def residuals_local(self, params, fGHz, acftype, RT_models, mv, vv_obs, theta_i, rvi, sand, clay, bulk, sst):
        d, c, s, l, omega = params

        ke = c * np.sqrt(rvi)

        db85 = Dobson85(clay=clay, sand=sand, mv=mv, freq=fGHz, temp=sst, bulk=bulk)
        eps2 = eps3 = np.array([db85.eps,], dtype=complex)

        # Create an instance of the S2RTR class
        with self._maybe_suppress_integration_warning():
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
        ks = []
        bs = []

        for idx, row in df_x.iterrows():

            try:
                vv = self.to_power(row['vv'])
                vh = self.to_power(row['vh'])
                vv_soil = np.nan
                vv_veg = np.nan

                denom = vh + vv
                if not np.isfinite(denom) or denom <= 0:
                    raise ValueError("Invalid VV/VH values for RVI computation")

                rvi = (4 * vh) / denom
                if not np.isfinite(rvi):
                    raise ValueError("Computed RVI is not finite")

                theta_i = row['angle']
                ssm = row['ssm']
                sst = row['sst']
                bbch = row['bbch']
                croptype = str(int(row['lc']))
                sand = row['sand']
                clay = row['clay']
                bulk = row['bulk']

                # Check if crop type has phenology data and inversion bounds available
                has_pheno_data = croptype in self.crop_bbch_k_b_coff
                has_bounds = croptype in bounds
                
                # Treat crops without phenology data or bounds as bare soil (like crops 30 and 34)
                # For these crops, we skip the inversion and just set vv_soil = vv (total backscatter)
                if croptype in ('30', '34') or not has_pheno_data or not has_bounds:
                    d = c = w = np.nan
                    # Use default surface roughness parameters for bare soil
                    s = 0.01  # Default RMS height in meters
                    l = 0.1   # Default correlation length in meters
                    vv_soil = vv
                    vv_veg = 0.0
                    height = (np.nan, np.nan, np.nan, np.nan, np.nan)
                else:
                    # Get bounds for this crop type
                    c_bound = bounds[croptype]['cbound']
                    s_bound = bounds[croptype]['sbound']
                    l_bound = bounds[croptype]['lbound']
                    w_bound = bounds[croptype]['wbound']
                    
                    height = self.estimate_crop_height_interp(croptype, bbch, rvi)
                    if not np.all(np.isfinite(height[:3])):
                        raise ValueError("Non-finite crop height estimate")

                    initial_guess = [height[1], c_bound[1], s_bound[1], l_bound[1], w_bound[1]]
                    lower_bounds = [height[0], c_bound[0], s_bound[0], l_bound[0], w_bound[0]]
                    upper_bounds = [height[2], c_bound[2], s_bound[2], l_bound[2], w_bound[2]]

                    res = least_squares(
                        self.residuals_local,
                        initial_guess,
                        args=(self.fGHz, self.acftype, self.models, ssm, vv, theta_i, rvi, sand, clay, bulk, sst),
                        bounds=(lower_bounds, upper_bounds),
                    )
                    d, c, s, l, w = res.x

                db85 = Dobson85(clay=clay, sand=sand, bulk=bulk, mv=ssm, freq=self.fGHz, t=sst)
                eps = np.array([db85.eps,], dtype=complex)
                pol_list = ['vv', 'hh', 'hv', 'vh']

                if self.models['RT_s'] == 'AIEM':
                    with self._maybe_suppress_integration_warning():
                        aiem_obj = AIEM0(frq_GHz=self.fGHz, acf=self.acftype, s=s, l=l, thi_deg=theta_i, ths_deg=theta_i, phi_deg=0, phs_deg=179.999, eps=eps)
                        sigma0 = aiem_obj.compute_sigma0(pol='vv', todB=False)
                    vv_soil = sigma0['vv'][0]
                elif self.models['RT_s'] == 'PRISM1':
                    with self._maybe_suppress_integration_warning():
                        prism0 = PRISM1(f=self.fGHz, theta_i=theta_i, eps=eps, s=s)
                        sig_0_top_full = prism0.calc_sigma(todB=False)
                    sig_0_top = dict(zip(pol_list, sig_0_top_full))
                    vv_soil = sig_0_top['vv'][0]
                elif self.models['RT_s'] == 'SMART':
                    with self._maybe_suppress_integration_warning():
                        smart = SMART(fGHz=self.fGHz, theta_deg=theta_i, s=s, eps=eps)
                        sig_0_full = smart.calc_sigma(todB=False)
                    sig_0_top = dict(zip(pol_list, sig_0_full))
                    vv_soil = sig_0_top['vv'][0]
                elif self.models['RT_s'] == 'SPM3D':
                    with self._maybe_suppress_integration_warning():
                        spm3d = SPM3D(fr=self.fGHz, sig=s, L=l, thi=theta_i, eps=eps)
                        sig_0_full = spm3d.calc_sigma(todB=False)
                    sig_0_top = dict(zip(pol_list, sig_0_full))
                    vv_soil = sig_0_top['vv'][0]
                elif self.models['RT_s'] == 'I2EM':
                    with self._maybe_suppress_integration_warning():
                        sp_map = {'exp': 1, 'gauss': 2, 'pow': 3}
                        sp = sp_map.get(self.acftype, 1)
                        xx = 1.5 if sp == 3 else 0.0
                        vv_dB, hh_dB, hv_dB, _ = I2EM_Bistat_model(
                            fr=self.fGHz,
                            sig=s,
                            L=l,
                            thi=theta_i,
                            ths=theta_i,
                            phs=179.999,
                            er=eps,
                            sp=sp,
                            xx=xx
                        )
                        vv_soil = self.to_power(vv_dB)
                else:
                    raise ValueError(f"Unsupported surface model: {self.models['RT_s']}")

                if croptype not in ('30', '34') and np.isfinite(vv_soil):
                    vv_veg = vv - vv_soil

            except Exception:
                # If an exception occurs, check if this is a crop without phenology data
                # In that case, treat it as bare soil
                croptype_str = str(int(row['lc'])) if 'lc' in row.index else 'unknown'
                has_pheno = croptype_str in self.crop_bbch_k_b_coff
                has_bounds_check = croptype_str in bounds
                
                if croptype_str in ('30', '34') or not has_pheno or not has_bounds_check:
                    # Treat as bare soil - preserve the data
                    d = c = w = np.nan
                    s = 0.01
                    l = 0.1
                    vv_soil = self.to_power(row['vv']) if 'vv' in row.index else np.nan
                    vv_veg = 0.0
                    rvi = np.nan
                    height = (np.nan, np.nan, np.nan, np.nan, np.nan)
                else:
                    # For crops that should have been inverted but failed, set everything to NaN
                    d = np.nan
                    c = np.nan
                    s = np.nan
                    l = np.nan
                    w = np.nan
                    vv_veg = np.nan
                    vv_soil = np.nan
                    rvi = np.nan
                    height = (np.nan, np.nan, np.nan, np.nan, np.nan)

            dvvs.append(d)
            cvvs.append(c)
            wvvs.append(w)
            vv_vegs.append(vv_veg)
            vv_soils.append(vv_soil)
            SSRs.append(s)
            SSRl.append(l)
            rvis.append(rvi)
            heights.append(height[1])
            ks.append(height[3])
            bs.append(height[4])

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
        df_x['k'] = ks
        df_x['b'] = bs

        return df_x
