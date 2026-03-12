# Getting Started with nitropulse Notebooks

This directory contains a series of Jupyter notebooks designed to guide you through the full `nitropulse` workflow, from raw data acquisition to machine learning model deployment.

## Prerequisites

Before running these notebooks, ensure you have:
1.  **Python 3.8+** installed.
2.  **Google Earth Engine (GEE)** account and the `earthengine` CLI authenticated (`earthengine authenticate`).
3.  **Installed dependencies**:
    ```bash
    pip install git+https://github.com/Morteza-Khazaei/AIEM.git
    pip install git+https://github.com/Morteza-Khazaei/SSRT.git
    pip install -e .
    ```

## Sequential Workflow

For the best experience, please run the notebooks in the following order:

### 1. Data Acquisition
*   **[01a_download_risma.ipynb](01a_download_risma.ipynb)**: Downloads ground truth soil moisture and temperature data from RISMA stations.
*   **[01b_download_s1.ipynb](01b_download_s1.ipynb)**: Requests Sentinel-1 SAR backscatter data from Google Earth Engine.
    > **Note**: S1 data is exported to your Google Drive. You must manually download the resulting CSVs into your workspace's `inputs/S1_CSV_files/` directory before proceeding to Stage 2.

### 2. Phenology Modeling
*   **[02_phenology_workflow.ipynb](02_phenology_workflow.ipynb)**: Combines RISMA and S1 data to calculate Growing Degree Days (GDD) and standardized BBCH growth stages. This stage estimates plant height and prepares the `pheno_df.csv` file.

### 3. Biophysical Inversion
*   **[03_biophysical_inversion.ipynb](03_biophysical_inversion.ipynb)**: Uses Radiative Transfer (RT) models (like AIEM or PRISM1) to separate soil and vegetation scattering. It retrieves key parameters like surface roughness (`s`) and vegetation water content (`c`).

### 4. Machine Learning
*   **[04_machine_learning.ipynb](04_machine_learning.ipynb)**: Trains Random Forest models using a holdout validation strategy (leaving out one year/station for testing). It evaluates performance using R², RMSE, and bias.

### 5. Validation & Visualization
*   **[05_validation_visualization.ipynb](05_validation_visualization.ipynb)**: Provides tools for generating publication-quality scatter plots and time-series comparisons between observed and predicted values.

## Tips for Success

- **Workspace Directory**: By default, `nitropulse` uses `~/.nitropulse`. You can change this in the notebooks by setting the `workspace_dir` parameter.
- **Kernel**: Ensure your Jupyter kernel is set to the environment where `nitropulse` is installed.
- **GEE Project**: Make sure your `gee_project_id` is correctly set in the download notebooks.
- **Manual Step**: Remember that the Sentinel-1 download (01b) is asynchronous. Check your GEE Tasks tab and Google Drive for the output files.
