# nitropulse

<p align="center">
  <a href="https://github.com/Morteza-Khazaei/nitropulse">
    <img src="https://raw.githubusercontent.com/Morteza-Khazaei/nitropulse/main/logo/nitropulse.png" alt="nitropulse logo" width="400" />
  </a>
</p>

<p align="center">
    <em>A precision tool for mapping soil moisture and biophysical parameters in agricultural landscapes using Sentinel-1 SAR and machine learning.</em>
</p>

<p align="center">
    <a href="https://github.com/Morteza-Khazaei/nitropulse/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
    <a href="#"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python version"></a>
    <a href="https://github.com/Morteza-Khazaei/nitropulse/actions/workflows/ci.yml"><img src="https://github.com/Morteza-Khazaei/nitropulse/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

---

## Overview

`nitropulse` is a comprehensive command-line tool that integrates remote sensing, radiative transfer modeling, and machine learning to map soil moisture and biophysical parameters at high spatial resolution (10m). The package processes data from RISMA ground stations and Sentinel-1 satellites, applies physics-based inversion algorithms, and trains Random Forest models for operational deployment on Google Earth Engine (GEE).

### Key Features

- **Multi-source Data Integration**: Seamlessly combines ground station measurements (RISMA) with Sentinel-1 SAR backscatter
- **Physics-Based Modeling**: Implements multiple radiative transfer models (AIEM, PRISM1, SPM3D, SMART, I2EM) for biophysical parameter retrieval
- **Phenology-Aware Processing**: Incorporates crop growth stages (BBCH scale) and Growing Degree Days (GDD) for improved accuracy
- **Machine Learning Pipeline**: Automated Random Forest training with holdout validation and performance metrics
- **GEE Deployment**: Direct deployment of trained models to Google Earth Engine for large-scale mapping
- **Flexible Workflow**: Run the complete pipeline or individual stages based on your needs

## Workflow Architecture

The `nitropulse` workflow consists of four main stages, as illustrated in the flowchart below:

<p align="center">
  <img src="https://raw.githubusercontent.com/Morteza-Khazaei/nitropulse/main/logo/Nitropulse_Flowchart.svg" alt="nitropulse workflow" width="100%" />
</p>

### Stage 1: 📥 Data Acquisition

Downloads and prepares multi-source datasets:

- **RISMA Ground Stations**: Soil moisture (SSM), soil temperature (SST), air temperature, and precipitation data
- **Sentinel-1 SAR**: VV/VH backscatter coefficients and incidence angles from Google Earth Engine
- **Temporal Alignment**: Automatic time-matching between satellite overpasses and ground measurements (±2 hour window)

### Stage 2: 🌱 Phenology Modeling

Calculates crop development indicators critical for accurate biophysical modeling:

- **Growing Degree Days (GDD)**: Computed from air and soil temperatures using crop-specific base temperatures
- **BBCH Growth Stages**: Maps GDD to standardized crop phenology stages (0-99 scale)
- **Crop Height Estimation**: Derives plant height from BBCH stages and Radar Vegetation Index (RVI)
- **Output**: `pheno_df.csv` containing SSM, BBCH, GDD, and RVI for each observation

### Stage 3: 🔄 Biophysical Inversion

Separates radar backscatter into soil and vegetation components using radiative transfer models:

- **RT Model Suite**: AIEM, PRISM1, SPM3D, SMART, I2EM for surface scattering; Water Cloud Model for canopy
- **Parameter Retrieval**: Inverts for soil roughness (s, l), vegetation water content (c), and soil backscatter (vvs)
- **Iterative Optimization**: Refines parameter bounds across observation groups (station, year, day, angle)
- **Dielectric Modeling**: Dobson model for soil permittivity calculation
- **Output**: `inv_df.csv` with retrieved biophysical parameters (SSM, s, l, c, vvs)

### Stage 4: 🤖 Machine Learning & Deployment

Trains and deploys Random Forest models for operational soil moisture mapping:

- **Holdout Validation**: One year per station reserved for testing to ensure spatial generalization
- **Sequential Prediction**: 
  1. Predict biophysical parameters (s, l, vvs) from VV, VH, angle, RVI, and temporal features
  2. Predict SSM from angle, vvs, s, l, and temporal features
- **Performance Metrics**: R², RMSE, unbiased RMSE (ubRMSE), and bias per crop type
- **GEE Deployment** (optional):
  - Converts Random Forest to decision tree strings
  - Uploads as FeatureCollections to GEE Assets
  - Enables large-scale prediction directly in Earth Engine
- **Prediction Stage**: Applies trained models to generate:
  - Regional SSM maps (10m resolution GeoTIFF or GEE Asset)
  - Station-level time series extracts
  - Validation plots and visualizations

## Installation

### Prerequisites

- Python 3.8 or higher
- Active Google Earth Engine account
- Google Cloud SDK (for GEE authentication)

### Step 1: Install Google Cloud SDK

The `earthengine` command requires the Google Cloud SDK. Follow the [official installation guide](https://cloud.google.com/sdk/docs/install), then initialize:

```bash
gcloud init
```

### Step 2: Authenticate with Google Earth Engine

```bash
earthengine authenticate
```

### Step 3: Install Required Dependencies

Install custom radiative transfer model packages from GitHub:

```bash
pip install git+https://github.com/Morteza-Khazaei/AIEM.git
pip install git+https://github.com/Morteza-Khazaei/SSRT.git
```

### Step 4: Install nitropulse

**From GitHub (recommended):**

```bash
pip install git+https://github.com/Morteza-Khazaei/nitropulse.git
```

**For Development:**

```bash
git clone https://github.com/Morteza-Khazaei/nitropulse.git
cd nitropulse
pip install -e .[test]
```

## Quick Start

You can run the complete `nitropulse` workflow using the CLI or explore the step-by-step Jupyter notebooks.

### Using the CLI

Run the full pipeline with a single command:

```bash
nitropulse run \
    --roi-asset-id "users/your-username/your-roi-asset" \
    --gee-project-id "your-gcp-project-id" \
    --workspace-dir "./my_workspace" \
    --start-date "2020-01-01" \
    --end-date "2023-12-31"
```

**Note**: Sentinel-1 data is exported to your Google Drive. After the export completes, manually download the folder to your workspace's `inputs/S1_CSV_files/` directory before proceeding to phenology and inversion stages.

### Using Jupyter Notebooks

For a detailed, interactive walkthrough of each stage, refer to the [notebooks/README.md](notebooks/README.md) guide. The notebooks are organized sequentially:

1.  **[01a_download_risma.ipynb](notebooks/01a_download_risma.ipynb)**: RISMA data acquisition.
2.  **[01b_download_s1.ipynb](notebooks/01b_download_s1.ipynb)**: Sentinel-1 GEE extraction.
3.  **[02_phenology_workflow.ipynb](notebooks/02_phenology_workflow.ipynb)**: GDD/BBCH modeling.
4.  **[03_biophysical_inversion.ipynb](notebooks/03_biophysical_inversion.ipynb)**: Radiative Transfer inversion.
5.  **[04_machine_learning.ipynb](notebooks/04_machine_learning.ipynb)**: Random Forest training.
6.  **[05_validation_visualization.ipynb](notebooks/05_validation_visualization.ipynb)**: Performance plotting.

## CLI Commands

### `nitropulse run`

Executes the complete workflow from data acquisition to model deployment.

**Required Arguments:**

- `--roi-asset-id`: GEE asset ID for your Region of Interest (e.g., `users/username/study_area`)
- `--gee-project-id`: Google Cloud Project ID linked to Earth Engine

**Optional Arguments:**

- `--workspace-dir`: Workspace directory (default: `~/.nitropulse`)
- `--stations`: Comma-separated station IDs (default: all RISMA stations)
- `--buffer-distance`: Buffer radius for S1 extraction in meters (default: `15`)
- `--start-date`: Start date in YYYY-MM-DD format (default: `2015-01-01`)
- `--end-date`: End date in YYYY-MM-DD format (default: `2024-01-01`)
- `--fghz`: Radar frequency in GHz (default: `5.4` for Sentinel-1 C-band)
- `--models`: RT models as JSON string (default: `{"RT_s": "PRISM1", "RT_c": "Diff"}`)
- `--acftype`: Autocorrelation function type for AIEM (default: `exp`)
- `--features`: Features for ML models as JSON list (default: `["SSM", "vvs", "s"]`)

**Example:**

```bash
nitropulse run \
    --roi-asset-id "users/john/saskatchewan_fields" \
    --gee-project-id "my-gee-project" \
    --workspace-dir "./sask_analysis" \
    --stations "RISMA_01,RISMA_02,RISMA_03" \
    --start-date "2021-01-01" \
    --end-date "2022-12-31" \
    --buffer-distance 20
```

### Individual Stage Commands

Run specific stages independently:

#### `nitropulse download-risma`

Downloads RISMA ground station data only.

```bash
nitropulse download-risma \
    --workspace-dir "./my_workspace" \
    --stations "RISMA_01,RISMA_02" \
    --start-date "2020-01-01" \
    --end-date "2023-12-31"
```

#### `nitropulse download-s1`

Downloads Sentinel-1 backscatter data from GEE.

```bash
nitropulse download-s1 \
    --roi-asset-id "users/username/roi" \
    --gee-project-id "my-project" \
    --workspace-dir "./my_workspace" \
    --buffer-distance 15 \
    --start-date "2020-01-01" \
    --end-date "2023-12-31"
```

#### `nitropulse phenology`

Runs phenology modeling (GDD and BBCH calculation).

```bash
nitropulse phenology --workspace-dir "./my_workspace"
```

#### `nitropulse inversion`

Executes biophysical parameter inversion.

```bash
nitropulse inversion \
    --workspace-dir "./my_workspace" \
    --fghz 5.4 \
    --models '{"RT_s": "AIEM", "RT_c": "WCM"}' \
    --acftype "exp"
```

#### `nitropulse modeling`

Trains Random Forest models and optionally deploys to GEE.

```bash
nitropulse modeling \
    --workspace-dir "./my_workspace" \
    --features '["SSM", "vvs", "s", "l"]' \
    --deploy-to-gee \
    --gee-project-id "my-project"
```

## Workspace Structure

`nitropulse` organizes all data in a structured workspace directory:

```
~/.nitropulse/                          # Default workspace location
├── config/                             # Auto-generated configuration files
│   ├── gdd/
│   │   ├── crop_base_temp.json        # Crop-specific base temperatures
│   │   ├── crop_bbch_k_b_coff.json    # BBCH-GDD mapping coefficients
│   │   └── crop_gdd_thresh.json       # GDD thresholds for BBCH stages
│   ├── inversion/
│   │   └── crop_inversion_bounds.json # Parameter bounds for RT inversion
│   └── risma/
│       └── stations_texture.json      # Soil texture data for stations
├── inputs/                             # Downloaded raw data
│   ├── RISMA_CSV_files/               # Ground station measurements
│   │   ├── RISMA_01_2020.csv
│   │   └── ...
│   └── S1_CSV_files/                  # Sentinel-1 backscatter data
│       ├── S1_VV_VH_2020.csv
│       └── ...
├── models/                             # Trained ML models
│   ├── SSM_model.joblib               # Soil moisture model
│   ├── vvs_model.joblib               # Soil backscatter model
│   ├── s_model.joblib                 # Surface roughness model
│   └── l_model.joblib                 # Correlation length model
├── outputs/                            # Processed results
│   ├── pheno_df.csv                   # Phenology modeling output
│   ├── inv_df.csv                     # Inversion results
│   ├── test_df.csv                    # Holdout validation predictions
│   └── metrics.json                   # Model performance metrics
└── figures/                            # Validation plots (if generated)
    ├── scatter_*.svg                  # Observed vs predicted scatter plots
    └── timeseries_*.svg               # Time series comparisons
```

## Configuration Files

`nitropulse` uses JSON configuration files for crop-specific parameters. These are automatically created in the workspace but can be customized:

### GDD Configuration (`config/gdd/`)

- **crop_base_temp.json**: Base temperatures (°C) for GDD calculation per crop
- **crop_gdd_thresh.json**: GDD thresholds for BBCH stage transitions
- **crop_bbch_k_b_coff.json**: Linear coefficients (k, b) for crop height estimation: `height = k × RVI + b`

### Inversion Configuration (`config/inversion/`)

- **crop_inversion_bounds.json**: Upper and lower bounds for RT model parameters (d, c, s, l, ω) per crop type

### RISMA Configuration (`config/risma/`)

- **stations_texture.json**: Soil texture properties (sand, clay, silt percentages) for each station

## Output Data Formats

### pheno_df.csv

Phenology modeling output with columns:

- `station`: Station ID
- `date`: Observation timestamp
- `year`, `doy`: Year and day of year
- `lc`: Land cover / crop type
- `ssm`: Surface soil moisture (m³/m³)
- `sst`: Soil surface temperature (°C)
- `prcp`: Precipitation (mm/day)
- `VV`, `VH`: Sentinel-1 backscatter (dB)
- `angle`: Incidence angle (degrees)
- `RVI`: Radar Vegetation Index
- `GDD`: Growing Degree Days
- `BBCH`: Crop growth stage (0-99)
- `PH`: Estimated plant height (m)

### inv_df.csv

Biophysical inversion output with additional columns:

- `s`: RMS surface roughness (cm)
- `l`: Correlation length (cm)
- `c`: Vegetation water content (kg/m²)
- `vvs`: Soil backscatter contribution (linear units)
- `RT_model`: Radiative transfer model used

### test_df.csv

Holdout validation predictions with columns:

- All columns from `inv_df.csv`
- `SSM_pred`, `vvs_pred`, `s_pred`, `l_pred`: Predicted values from Random Forest models

## Advanced Usage

### Custom RT Model Configuration

Specify different radiative transfer models for surface and canopy:

```bash
nitropulse inversion \
    --models '{"RT_s": "AIEM", "RT_c": "WCM"}' \
    --acftype "gauss"
```

Available surface models: `AIEM`, `PRISM1`, `SPM3D`, `SMART`, `I2EM`  
Available canopy models: `WCM` (Water Cloud Model), `Diff` (Differential)

### Deploying to Google Earth Engine

After training models, deploy them to GEE for large-scale prediction:

```bash
nitropulse modeling \
    --workspace-dir "./my_workspace" \
    --deploy-to-gee \
    --gee-project-id "my-gee-project" \
    --gee-asset-prefix "users/username/nitropulse_models"
```

This converts Random Forest models to decision tree strings and uploads them as FeatureCollections to your GEE assets.

### Generating Validation Plots

Use the built-in validation utilities to create publication-quality figures:

```python
from nitropulse.utils.validation import plot_scatter, plot_timeseries

# Load your data
import pandas as pd
obs_df = pd.read_csv("~/.nitropulse/outputs/pheno_df.csv")
test_df = pd.read_csv("~/.nitropulse/outputs/test_df.csv")

# Create scatter plot
fig = plot_scatter(
    obs_df, test_df,
    target="ssm",
    save="ssm_validation",
    workspace_dir="~/.nitropulse"
)

# Create time series plot
fig = plot_timeseries(
    obs_df, test_df,
    station="RISMA_01",
    year=2021,
    target="ssm",
    save="ssm_timeseries",
    workspace_dir="~/.nitropulse"
)
```

Figures are saved as SVG format in the `figures/` subdirectory for high-quality publication.

## Performance Considerations

- **Processing Time**: The complete workflow can take several hours depending on:
  - Number of stations and temporal extent
  - RT model complexity (AIEM is slower than PRISM1)
  - Number of Random Forest trees and depth
  
- **Memory Requirements**: 
  - Minimum 8 GB RAM recommended
  - 16+ GB for large regions or long time series
  
- **GEE Quotas**: 
  - Sentinel-1 exports are subject to GEE computational limits
  - Large exports may need to be split into smaller temporal chunks

## Troubleshooting

### Common Issues

**1. GEE Authentication Errors**

```bash
# Re-authenticate with Earth Engine
earthengine authenticate --force
```

**2. Sentinel-1 Export Not Completing**

- Check your Google Drive for the export task status
- Large exports may take hours; monitor in the GEE Code Editor Tasks tab
- Reduce temporal extent or spatial coverage if exports fail

**3. Missing Configuration Files**

Configuration files are auto-generated on first run. If missing:

```bash
# Re-run setup to regenerate configs
nitropulse run --roi-asset-id "your/asset" --gee-project-id "your-project"
```

**4. Inversion Convergence Issues**

If RT inversion fails to converge:
- Try a different RT model (e.g., switch from AIEM to PRISM1)
- Adjust parameter bounds in `config/inversion/crop_inversion_bounds.json`
- Check for outliers in input data

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the Repository**: Create your own fork of the project
2. **Create a Feature Branch**: `git checkout -b feature/your-feature-name`
3. **Make Changes**: Implement your feature or bug fix
4. **Add Tests**: Ensure new code is covered by tests
5. **Commit Changes**: `git commit -m 'Add descriptive commit message'`
6. **Push to Branch**: `git push origin feature/your-feature-name`
7. **Open Pull Request**: Submit a PR with a clear description of changes

### Development Setup

```bash
git clone https://github.com/Morteza-Khazaei/nitropulse.git
cd nitropulse
pip install -e .[test]
pytest tests/
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use `nitropulse` in your research, please cite:

```bibtex
@software{nitropulse2024,
  title={{nitropulse}: A precision tool for mapping soil moisture and biophysical parameters using Sentinel-1 SAR and machine learning},
  author={Khazaei, Morteza},
  year={2024},
  url={https://github.com/Morteza-Khazaei/nitropulse},
  version={0.1.0},
  license={Apache-2.0}
}
```

## Acknowledgments

- **RISMA Network**: For providing high-quality ground station data
- **Google Earth Engine**: For Sentinel-1 data access and computational infrastructure
- **Radiative Transfer Models**: AIEM, PRISM1, SPM3D implementations adapted from published literature

## Contact

For questions, issues, or collaboration inquiries:

- **GitHub Issues**: [https://github.com/Morteza-Khazaei/nitropulse/issues](https://github.com/Morteza-Khazaei/nitropulse/issues)
- **Email**: [morteza.khazaei@usask.ca](mailto:morteza.khazaei@usask.ca)

---
