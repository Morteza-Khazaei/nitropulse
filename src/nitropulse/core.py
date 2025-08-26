import os
import argparse
import json
import shutil
import pkg_resources

from nitropulse.risma import RismaData
from nitropulse.radar import S1Data
from nitropulse.pheno import BBCH
from nitropulse.inverse import Inverse
from nitropulse.ensemble_rf import RegressionRF



def run_download(args):
    """
    Download RISMA and Sentinel-1 data.
    """
    print("🚀 Starting data download...")
    print(f"Workspace: {args.workspace_dir}")
    print(f"Stations: {', '.join(args.stations)}")
    print(f"Buffer distance: {args.buffer_distance} m")
    print(f"Date range: {args.start_date} to {args.end_date}")

    # Initialize data handlers
    risma = RismaData(args.workspace_dir)
    s1 = S1Data(args.workspace_dir)

    print("⬇️  Downloading RISMA data...")
    risma.download_risma_data(
        out_dir=os.path.join(args.workspace_dir, 'inputs', 'RISMA_CSV_files'),
        stations=args.stations,
        parameters=['Air Temp', 'Soil temperature', 'Soil Moisture'],
        sensors='average',
        depths=['0 to 5 cm', '5 cm'],
        start_date=args.start_date,
        end_date=args.end_date
    )

    print("⬇️  Downloading S1 data...")
    s1.download_S1_data(
        stations=args.stations,
        buffer_distance=args.buffer_distance,
        start_date=args.start_date,
        end_date=args.end_date,
        roi_asset_id=args.roi_asset_id,
        gee_project_id=args.gee_project_id
    )
    print("✅ Data download completed!")


def run_phenology(args):
    """
    Run the phenology model.
    """
    print("🌱 Running phenology model...")
    bbch = BBCH(args.workspace_dir)
    pheno_df = bbch.run()

    output_file = os.path.join(args.workspace_dir, 'outputs', 'pheno_df.csv')
    print(f"💾 Saving phenology results to {output_file}.")
    pheno_df.to_csv(output_file, index=False)
    print("✅ Phenology model completed!")


def run_inversion(args):
    """
    Run the inversion algorithm.
    """
    print("🔄 Running inversion...")
    try:
        models_dict = json.loads(args.models)
    except Exception as e:
        print(f"❌ Error parsing models JSON: {e}")
        return

    pheno_df_path = os.path.join(args.workspace_dir, 'outputs', 'pheno_df.csv')
    if not os.path.exists(pheno_df_path):
        print(f"Phenology output file not found at {pheno_df_path}")
        return
    
    import pandas as pd
    pheno_df = pd.read_csv(pheno_df_path)

    inv = Inverse(args.workspace_dir, fGHz=args.fghz, models=models_dict, acftype=args.acftype)
    inv_df = inv.run(pheno_df)

    output_file = os.path.join(args.workspace_dir, 'outputs', 'inv_df.csv')
    print(f"💾 Saving inversion results to {output_file}.")
    inv_df.to_csv(output_file, index=False)
    print("✅ Inversion completed!")


def run_modeling(args):
    """
    Train ensemble models and deploy to GEE.
    """
    print('🔄 Running ensemble training...')
    inv_df_path = os.path.join(args.workspace_dir, 'outputs', 'inv_df.csv')
    if not os.path.exists(inv_df_path):
        print(f"Inversion output file not found at {inv_df_path}")
        return
        
    import pandas as pd
    inv_df = pd.read_csv(inv_df_path)
    
    ensrf = RegressionRF(args.workspace_dir, inv_df)
    rf_models = ensrf.run(vars=list(args.features))
    print('✅ Ensemble training completed!')

    if args.gee_project_id:
        print('🔄 Uploading ensemble models to GEE...')
        ensrf.upload_rf_to_gee(rf_models, args.gee_project_id, save_dectree=False)
        print('✅ Ensemble models uploaded to GEE!')
    else:
        print('⚠️  No GEE project ID provided, skipping upload to GEE.')

def run_workflow(args):
    """
    Run the complete inversion workflow.
    """
    print("🚀 Starting full inversion workflow...")
    run_download(args)
    run_phenology(args)
    run_inversion(args)
    run_modeling(args)
    print("✅ Full workflow completed successfully!")


def setup_workspace(workspace_dir):
    """
    Ensures the workspace directory and its necessary subdirectories and
    configuration files exist. This is the first step.
    """
    print(f"Verifying workspace at: {workspace_dir}")

    # Create standard subdirectories
    subdirs = ['inputs', 'outputs', 'models', 'config/gdd']
    for subdir in subdirs:
        os.makedirs(os.path.join(workspace_dir, subdir), exist_ok=True)

    # List of config files to copy from the package to the workspace
    # Assumes they are in a 'config' directory inside the 'nitropulse' package
    config_templates = [
        'config/gdd/crop_base_temp.json',
        'config/gdd/crop_gdd_thresh.json',
    ]

    for template_path in config_templates:
        dest_path = os.path.join(workspace_dir, template_path)
        if not os.path.exists(dest_path):
            try:
                # Use pkg_resources to find the file within the installed package
                source_path = pkg_resources.resource_filename('nitropulse', template_path)
                if os.path.exists(source_path):
                    print(f"Initializing config file: {dest_path}")
                    shutil.copy(source_path, dest_path)
            except (ModuleNotFoundError, KeyError):
                # This can happen in a dev environment or if the resource doesn't exist.
                print(f"⚠️ Could not find packaged resource for '{template_path}'. Please ensure it exists.")
    
    print("✅ Workspace verification complete.")

def create_parser():
    """
    Creates and configures the argument parser for the CLI.
    """
    risma_stations = ['RISMA_MB1', 'RISMA_MB2', 'RISMA_MB3', 'RISMA_MB4', 'RISMA_MB5',
                      'RISMA_MB6', 'RISMA_MB7', 'RISMA_MB8', 'RISMA_MB9', 'RISMA_MB10',
                      'RISMA_MB11', 'RISMA_MB12', 'RISMA_MB13']
    
    home_dir = os.path.expanduser("~")
    default_workspace = os.path.join(home_dir, '.nitropulse')

    parser = argparse.ArgumentParser(
        description='A command-line interface for running SAR backscatter inversion models.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- Run All Command ---
    run_parser = subparsers.add_parser('run', help='Run the full workflow from download to modeling.')
    run_parser.add_argument('--workspace-dir', default=default_workspace, help='Workspace directory for all data and outputs.')
    run_parser.add_argument('--stations', nargs='*', default=risma_stations, help='List of station IDs to process.')
    run_parser.add_argument('--buffer-distance', type=int, default=15, help='Buffer distance for S1 data (meters).')
    run_parser.add_argument('--start-date', default='2010-01-01', help='Start date for data retrieval (YYYY-MM-DD).')
    run_parser.add_argument('--end-date', default='2024-01-01', help='End date for data retrieval (YYYY-MM-DD).')
    run_parser.add_argument('--roi-asset-id', required=True, help='GEE asset ID for the Region of Interest (e.g., users/user/roi).')
    run_parser.add_argument('--gee-project-id', required=True, help='Google Earth Engine project ID.')
    run_parser.add_argument('--fghz', type=float, default=5.4, help='Frequency in GHz for inversion.')
    run_parser.add_argument('--models', default='{"RT_s": "PRISM1", "RT_v": "Diff"}', help='RT models in JSON format.')
    run_parser.add_argument('--acftype', default='exp', help='ACF type for AIEM model.')
    run_parser.add_argument('--features', nargs='*', default=['SSM', 'vvs', 's'], help='Feature list for ML models.')
    run_parser.set_defaults(func=run_workflow)

    # Download command
    download_parser = subparsers.add_parser('download', help='Download RISMA and Sentinel-1 data')
    download_parser.add_argument('--workspace-dir', default=default_workspace, help='Workspace directory')
    download_parser.add_argument('--stations', nargs='*', default=risma_stations, help='List of station IDs')
    download_parser.add_argument('--buffer-distance', type=int, default=15, help='Buffer distance for S1 data')
    download_parser.add_argument('--start-date', default='2010-01-01', help='Start date for data')
    download_parser.add_argument('--end-date', default='2024-01-01', help='End date for data')
    download_parser.add_argument('--roi-asset-id', required=True, help='GEE asset ID for the Region of Interest')
    download_parser.add_argument('--gee-project-id', required=True, help='Google Earth Engine project ID for S1 data download')
    download_parser.set_defaults(func=run_download)

    # Phenology command
    phenology_parser = subparsers.add_parser('phenology', help='Run the phenology model')
    phenology_parser.add_argument('--workspace-dir', default=default_workspace, help='Workspace directory')
    phenology_parser.set_defaults(func=run_phenology)

    # Inversion command
    inversion_parser = subparsers.add_parser('inversion', help='Run the inversion algorithm')
    inversion_parser.add_argument('--workspace-dir', default=default_workspace, help='Workspace directory')
    inversion_parser.add_argument('--fghz', type=float, default=5.4, help='Frequency in GHz')
    inversion_parser.add_argument('--models', default='{"RT_s": "PRISM1", "RT_v": "Diff"}', help='RT models in JSON format')
    inversion_parser.add_argument('--acftype', default='exp', help='ACF type for AIEM model')
    inversion_parser.set_defaults(func=run_inversion)

    # Modeling command
    modeling_parser = subparsers.add_parser('modeling', help='Train ensemble models and deploy to GEE')
    modeling_parser.add_argument('--workspace-dir', default=default_workspace, help='Workspace directory')
    modeling_parser.add_argument('--features', nargs='*', default=['SSM', 'vvs', 's'], help='Feature list')
    modeling_parser.add_argument('--gee-project-id', help='Google Earth Engine project ID for model deployment')
    modeling_parser.set_defaults(func=run_modeling)

    return parser


def main():
    """
    A command-line interface for running SAR backscatter inversion models.
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Ensure the workspace is set up correctly before running any command.
    setup_workspace(args.workspace_dir)

    args.func(args)


if __name__ == "__main__":
    main()