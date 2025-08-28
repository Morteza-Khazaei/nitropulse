# nitropulse

<p align="center">
  <a href="https://github.com/Morteza-Khazaei/nitropulse">
    <!-- Add your logo here -->
    <img src="https://raw.githubusercontent.com/Morteza-Khazaei/nitropulse/main/logo/nitropulse.png" alt="nitropulse logo" width="400"/>
  </a>
</p>

<p align="center">
    <em>A precision tool for mapping nitrous oxide (N₂O) emission pulses in agricultural landscapes.</em>
</p>

<p align="center">
    <a href="https://github.com/Morteza-Khazaei/nitropulse/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
    <a href="#"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python version"></a>
    <a href="https://github.com/Morteza-Khazaei/nitropulse/actions"><img src="https://github.com/Morteza-Khazaei/nitropulse/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</p>

---

## Overview

`nitropulse` is a command-line tool designed to process remote sensing and ground data to estimate key agricultural parameters. It integrates data from RISMA ground stations and Sentinel-1 satellites, runs biophysical models, and trains machine learning ensembles to map soil moisture and other properties at scale.

The workflow consists of four main stages:
1.  **Data Acquisition**: Downloads and prepares soil moisture/temperature data from RISMA stations and Sentinel-1 backscatter data from Google Earth Engine (GEE).
2.  **Phenology Modeling**: Calculates Growing Degree Days (GDD) and uses them to model crop growth stages (BBCH scale), a critical input for radiative transfer modeling.
3.  **Biophysical Inversion**: Separates the total backscatter signal into soil and vegetation components using radiative transfer models (e.g., AIEM, PRISM1) to retrieve physical parameters like soil roughness.
4.  **Machine Learning & Deployment**: Trains Random Forest models to estimate soil moisture and other variables from backscatter data and deploys these models to GEE for large-scale application.

## Installation

### Dependencies
- Python 3.8+
- An active Google Earth Engine account.

### 1. Install Google Cloud SDK
The `earthengine` command is part of the Google Cloud SDK. Before you can authenticate, you must install it by following the official Google Cloud SDK installation instructions.

After installation, initialize the SDK by running `gcloud init`.

### 2. Authenticate with Google Earth Engine
Once the Google Cloud SDK is installed, authenticate your machine with GEE.
```bash
earthengine authenticate
```

### 3. Install Dependencies from GitHub
`nitropulse` relies on several custom packages that must be installed directly from their GitHub repositories.
```bash
pip install git+https://github.com/Morteza-Khazaei/AIEM.git
pip install git+https://github.com/Morteza-Khazaei/SSRT.git
```

### 4. Install `nitropulse`
Install the latest version of `nitropulse` from GitHub.
```bash
pip install git+https://github.com/Morteza-Khazaei/nitropulse.git
```

### For Development
If you want to contribute to or modify the package:
```bash
git clone https://github.com/Morteza-Khazaei/nitropulse.git
cd nitropulse
pip install -e .[test]
```

## Quick Start
Run the entire workflow with a single command. This will download data, run all models, and save the outputs to a workspace directory (default: `~/.nitropulse`).

**Note**: The Sentinel-1 download process exports data to your Google Drive. You may need to manually download the exported folder to your workspace.

```bash
nitropulse run \
    --roi-asset-id "your/gee/asset/path" \
    --gee-project-id "your-gcp-project-id"
```

## CLI Commands
`nitropulse` provides a modular, command-based interface. You can run the full workflow or individual steps. For a full list of options for any command, use the `--help` flag (e.g., `nitropulse run --help`).

### `nitropulse run`
Runs the complete workflow from data download to model deployment. This is the recommended command for most users.

**Required Arguments:**
- `--roi-asset-id`: Your GEE asset ID for the Region of Interest (e.g., `users/username/roi_asset`).
- `--gee-project-id`: Your Google Cloud Project ID associated with Earth Engine.

### Other Commands
- `nitropulse download-risma`: Downloads only the RISMA ground station data.
- `nitropulse download-s1`: Downloads only the Sentinel-1 backscatter data from GEE.
- `nitropulse phenology`: Runs the phenology model.
- `nitropulse inversion`: Runs the inversion algorithm.
- `nitropulse modeling`: Trains ensemble models and optionally deploys them to GEE.

## Workspace Structure
`nitropulse` creates a workspace directory to store all configurations, inputs, and outputs. By default, this is located at `~/.nitropulse`.

```
~/.nitropulse/
├── config/                 # Configuration files for models (auto-generated)
│   ├── gdd/
│   └── inversion/
├── inputs/                 # Raw data downloaded by the tool
│   ├── RISMA_CSV_files/
│   └── S1_CSV_files/
├── models/                 # Trained machine learning models (*.joblib)
└── outputs/                # Processed data and results
│   ├── pheno_df.csv
│   └── inv_df.csv
```

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a new feature branch (`git checkout -b feature/your-feature-name`)
3. Make your changes and commit them (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature-name`)
5. Open a Pull Request

## License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Citation
If you use this package in your research, please cite:

```bibtex
@software{nitropulse,
  title={{nitropulse}: A precision tool for mapping nitrous oxide (N₂O) emission pulses in agricultural landscapes},
  author={Khazaei, Morteza},
  year={2024},
  url={https://github.com/Morteza-Khazaei/nitropulse},
  version={0.1.0}
}
```