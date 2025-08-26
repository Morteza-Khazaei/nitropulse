# SAR Backscatter Inversion Package

A Python package for estimating soil and vegetation parameters from Sentinel-1 radar data using ground-based measurements from RISMA (Real-time In-Situ Soil Monitoring for Agriculture) networks.

## Overview

The inversion workflow consists of the following main steps:

### 1. Data Acquisition and Preparation
- **RISMA Ground Measurements**:
  - Downloads and prepares soil moisture and temperature data from RISMA stations.
  - Stores all necessary CSV files in the `./assets/inputs/...` directory.
- **Sentinel-1 Backscatter Data**:
  - Retrieves Sentinel-1 backscatter data from Google Earth Engine (GEE).
  - Applies spatial buffering around each RISMA station to a reliable time series of backscatter data for each station.

### 2. Growing Degree Days (GDD) and Phenology Modeling
- **GDD Calculation**:
  - Calculates accumulated Growing Degree Days (GDD) from RISMA temperature data.
- **BBCH Scale**:
  - Uses GDD to compute the BBCH scale, generating a time series of crop growth stages.
  - Estimates crop height per day of year (DOY), a critical parameter for vegetation radiative transfer modeling (RTM).

### 3. Inversion Process
- **Backscatter Separation**:
  - Separates total backscatter from Sentinel-1 images into soil and vegetation components.
- **Radiative Transfer Models**:
  - Uses vegetation RTM and crop height data to accurately estimate soil and vegetation parameters.

### 4. Machine Learning and Deployment
- **Ensemble Model Training**:
  - Trains and tests ensemble models to estimate:
    - Bare soil backscatter
    - Soil roughness
    - Soil moisture
- **Google Earth Engine Deployment**:
  - Deploys trained models to the GEE platform for large-scale soil moisture estimation using Sentinel-1 backscatter images.

## Installation

### Dependencies

The package automatically installs the following dependencies:

- Python 3.8+
- Google Earth Engine Python API
- NumPy, Pandas, SciPy
- scikit-learn
- RISMA *developed during this project*
- pyAIEM *developed during this project*
- SSRT *developed during this project*

### Install from GitHub

```bash
pip install git+https://github.com/Morteza-Khazaei/AIEM.git
pip install git+https://github.com/Morteza-Khazaei/SSRT.git
pip install git+https://github.com/Morteza-Khazaei/inversion.git
```

### For Development

If you want to contribute or modify the package:

```bash
git clone https://github.com/Morteza-Khazaei/inversion.git
cd inversion
pip install -e .
```

### Verify Installation

After installation, verify the package is working:

```bash
python -c "import inversion; print('Installation successful!')"
```

## Quick Start

### 1. Install the Package

```bash
pip install git+https://github.com/Morteza-Khazaei/inversion.git
```

### 2. Set up Google Earth Engine

```bash
earthengine authenticate
```

### 3. Basic Usage

Run the complete inversion workflow with default settings:

```bash
python -m inversion --auto-download --gee-project-id your-gcp-project
```

Or if you have the CLI script directly:

```bash
python inversion.py --auto-download --gee-project-id your-gcp-project
```

### 4. Process Specific Stations

```bash
python -m inversion --stations MB1 MB2 MB3 \
                   --auto-download \
                   --gee-project-id your-project-id
```

### 5. Custom Date Range and Parameters

```bash
python -m inversion --workspace-dir ./data \
                   --auto-download \
                   --stations MB1 MB2 MB3 MB4 \
                   --buffer-distance 20 \
                   --start-date 2020-01-01 \
                   --end-date 2023-12-31 \
                   --gee-project-id your-project-id \
                   --features SSM vvs s \
                   --fghz 5.4
```

## Command Line Arguments

### Core Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--workspace-dir` | str | `./assets` | Workspace directory for data and outputs |
| `--auto-download` | flag | False | Auto download Sentinel-1 data via GEE |
| `--stations` | list | All RISMA stations | Station IDs to process (space-separated) |
| `--gee-project-id` | str | None | Google Earth Engine project ID |

### Data Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--buffer-distance` | int | 15 | Buffer distance for S1 data (meters) |
| `--start-date` | str | `2010-01-01` | Start date (YYYY-MM-DD) |
| `--end-date` | str | `2024-01-01` | End date (YYYY-MM-DD) |
| `--roi-asset-id` | str | None | Region of interest asset ID in GEE |

### Inversion Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--fghz` | float | 5.4 | Frequency in GHz for inversion |
| `--models` | str | `{"RT_s": "PRISM1", "RT_v": "Diff"}` | RT models (JSON format) |
| `--acftype` | str | `exp` | ACF type for AIEM model |
| `--features` | list | `['SSM', 'vvs', 's']` | Feature list for ML models |

## Available RISMA Stations

The package includes 13 RISMA stations in Manitoba, Canada:

- `RISMA_MB1` through `RISMA_MB13`

You can specify individual stations or use the default (all stations):

```bash
# Process specific stations
python -m inversion --stations MB1 MB5 MB10

# Process all stations (default)
python -m inversion --auto-download
```

## Workflow Steps

### 1. Data Preparation
- Downloads RISMA ground-truth data (soil moisture, temperature)
- Applies spatial buffering around station locations and retrieves Sentinel-1 backscatter data from Google Earth Engine

### 2. Phenology Modeling
- Runs BBCH phenology model to determine crop growth stages
- Outputs phenology data to `outputs/pheno_df.csv`

### 3. Inversion Process
- Applies radiative transfer models (PRISM1, Diff)
- Uses AIEM for surface scattering modeling
- Estimates soil and vegetation parameters
- Outputs inversion results to `outputs/inv_df.csv`

### 4. Machine Learning
- Trains Random Forest ensemble models
- Uses specified features for parameter estimation
- Uploads trained models to Google Earth Engine

## Output Files

The package generates several output files in the `outputs/` directory:

- `pheno_df.csv`: Phenology model results with crop growth stages
- `inv_df.csv`: Inversion results with estimated parameters
- Various intermediate data files and model artifacts

## Configuration

### Custom Models

You can specify different radiative transfer models:

```bash
python -m inversion --models '{"RT_s": "PRISM2", "RT_v": "WCM"}'
```

### Feature Selection

Customize the features used for machine learning:

```bash
python -m inversion --features SSM VWC roughness lai
```

## Google Earth Engine Setup

1. Create a Google Cloud Project
2. Enable the Earth Engine API
3. Authenticate using `earthengine authenticate`
4. Provide your project ID when running the CLI

```bash
python -m inversion --gee-project-id your-gcp-project-id
```

## Troubleshooting

### Common Issues

**Authentication Error**: Ensure you've authenticated with Google Earth Engine:
```bash
earthengine authenticate
```

**Memory Issues**: For large datasets, consider:
- Processing fewer stations at once
- Reducing the date range
- Increasing buffer distance cautiously

**Missing Data**: Check that:
- Station IDs are correct
- Date ranges overlap with available data
- GEE project has necessary permissions

### Debug Mode

For debugging, you can inspect intermediate outputs in the workspace directory structure:

```
workspace/
├── inputs/
│   ├── RISMA_CSV_SSM_SST_AirT_2015_2023_new/
│   └── S1_data/
├── outputs/
│   ├── pheno_df.csv
│   └── inv_df.csv
└── models/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{sar_inversion_package,
  title={SAR Backscatter Inversion Package},
  author={Morteza Khazaei},
  year={2024},
  url={https://github.com/Morteza-Khazaei/inversion}
}
```

## Support

For support and questions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation wiki

---

**Note**: This package is designed for research purposes and requires appropriate permissions for accessing Google Earth Engine and RISMA data networks.