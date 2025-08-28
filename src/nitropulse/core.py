import os
import argparse
import json
import shutil
import pkg_resources
import sys
import joblib

from nitropulse.risma import RismaData
from nitropulse.radar import S1Data
from nitropulse.pheno import BBCH
from nitropulse.inverse import Inverse
from nitropulse.ensemble_rf import RegressionRF


def run_download_risma(args):
    """
    Download RISMA data.
    """
    print("🚀 Starting RISMA data download...")
    print(f"Workspace: {args.workspace_dir}")
    print(f"Stations: {', '.join(args.stations)}")
    print(f"Date range: {args.start_date} to {args.end_date}")

    risma = RismaData(args.workspace_dir)

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
    print("✅ RISMA data download completed!")


def run_download_s1(args):
    """
    Download Sentinel-1 data via GEE.
    """
    print("🚀 Starting Sentinel-1 data download...")
    print(f"Workspace: {args.workspace_dir}")
    print(f"Stations: {', '.join(args.stations)}")
    print(f"Buffer distance: {args.buffer_distance} m")
    print(f"Date range: {args.start_date} to {args.end_date}")

    s1 = S1Data(args.workspace_dir)

    print("⬇️  Downloading S1 data...")
    s1.download_S1_data(
        stations=args.stations,
        buffer_distance=args.buffer_distance,
        start_date=args.start_date,
        end_date=args.end_date,
        roi_asset_id=args.roi_asset_id,
        gee_project_id=args.gee_project_id
    )
    print("✅ S1 data download completed!")


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
        print(f"❌ Phenology output file not found at {pheno_df_path}")
        print("Please run the 'phenology' or 'run' command first.")
        sys.exit(1)

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
        print(f"❌ Inversion output file not found at {inv_df_path}")
        print("Please run the 'inversion' or 'run' command first.")
        sys.exit(1)

    import pandas as pd
    inv_df = pd.read_csv(inv_df_path)
    
    ensrf = RegressionRF(args.workspace_dir, inv_df)
    rf_models = ensrf.run(vars=list(args.features))
    print('✅ Ensemble training completed!')

    print("💾 Saving trained models to workspace...")
    models_dir = os.path.join(args.workspace_dir, 'models')
    for var, model in rf_models.items():
        model_filename = f"rf_model_{var}.joblib"
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(model, model_path)
        print(f"   -> Saved {model_filename}")
    print("✅ Models saved successfully!")

    if args.gee_project_id:
        print('🔄 Uploading ensemble models to GEE...')
        ensrf.upload_rf_to_gee(rf_models, args.gee_project_id, save_dectree=False)
        print('✅ Ensemble models uploaded to GEE!')
    else:
        print('⚠️  No GEE project ID provided, skipping upload to GEE.')

def setup_workspace(workspace_dir):
    """
    Ensures the workspace directory and its necessary subdirectories and
    configuration files exist. This is the first step.
    """
    print(f"Verifying workspace at: {workspace_dir}")

    # Create standard subdirectories for inputs, outputs, models, and configs
    subdirs = ['inputs', 'outputs', 'models', 'config/gdd', 'config/inversion']
    for subdir in subdirs:
        os.makedirs(os.path.join(workspace_dir, subdir), exist_ok=True)

    # List of config files to copy from the package to the workspace
    # These files are essential for the application to run and are expected
    # to be included in the package data upon installation.
    config_templates = [
        'config/gdd/crop_base_temp.json',
        'config/gdd/crop_gdd_thresh.json',
        'config/gdd/crop_bbch_k_b_coff.json',
        'config/inversion/crop_inversion_bounds.json',
    ]

    for template_path in config_templates:
        dest_path = os.path.join(workspace_dir, template_path)
        # If a config file doesn't exist in the workspace, copy it from the package
        if not os.path.exists(dest_path):
            try:
                # Use pkg_resources to find the file within the installed package
                source_path = pkg_resources.resource_filename('nitropulse', template_path)
                print(f"Initializing config file: {os.path.basename(dest_path)}")
                shutil.copy(source_path, dest_path) # This can raise FileNotFoundError
            except (ModuleNotFoundError, KeyError, FileNotFoundError):
                print(f"❌ Error: Packaged config file '{template_path}' not found.")
                print("   This is a critical error. It can be caused by two things:")
                print("     1. An incomplete installation of the 'nitropulse' package.")
                print("     2. (For developers) The 'config' directory is not located inside 'src/nitropulse/'.")
                print("   Please verify your project structure and installation.")
                sys.exit(1)
    
    print("✅ Workspace verification complete.")

def run_full_workflow(args):
    """Runs the complete workflow."""
    setup_workspace(args.workspace_dir)
    
    print("\n--- Step 1 of 4: Data Download ---")
    run_download_risma(args)
    run_download_s1(args)
    
    print("\n--- Step 2 of 4: Phenology Modeling ---")
    run_phenology(args)
    
    print("\n--- Step 3 of 4: Inversion ---")
    run_inversion(args)
    
    print("\n--- Step 4 of 4: Model Training and Deployment ---")
    run_modeling(args)
    
    print("\n✅ Full workflow completed successfully!")

def download_risma_command(args):
    setup_workspace(args.workspace_dir)
    run_download_risma(args)

def download_s1_command(args):
    setup_workspace(args.workspace_dir)
    run_download_s1(args)

def phenology_command(args):
    setup_workspace(args.workspace_dir)
    run_phenology(args)

def inversion_command(args):
    setup_workspace(args.workspace_dir)
    run_inversion(args)

def modeling_command(args):
    setup_workspace(args.workspace_dir)
    run_modeling(args)

def main():
    """
    A command-line interface for running SAR backscatter inversion models.
    """
    parser = argparse.ArgumentParser(
        description="nitropulse: A precision tool for mapping nitrous oxide (N₂O) emission pulses in agricultural landscapes.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    home_dir = os.path.expanduser("~")
    default_workspace = os.path.join(home_dir, '.nitropulse')
    risma_stations = ['RISMA_MB1', 'RISMA_MB2', 'RISMA_MB3', 'RISMA_MB4', 'RISMA_MB5',
                      'RISMA_MB6', 'RISMA_MB7', 'RISMA_MB8', 'RISMA_MB9', 'RISMA_MB10',
                      'RISMA_MB11', 'RISMA_MB12', 'RISMA_MB13']
    default_models_str = '{"RT_s": "PRISM1", "RT_v": "Diff"}'

    # Parent parser for common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--workspace-dir', default=default_workspace, help=f'Workspace directory for all data and outputs.\n(default: {default_workspace})')

    # --- Download RISMA command ---
    parser_download_risma = subparsers.add_parser('download-risma', help='Downloads RISMA ground station data.', parents=[parent_parser])
    parser_download_risma.add_argument('--stations', nargs='+', default=risma_stations, help=f"List of station IDs to process.\n(default: all stations)")
    parser_download_risma.add_argument('--start-date', default='2015-01-01', help='Start date for data retrieval (YYYY-MM-DD).\n(default: 2015-01-01)')
    parser_download_risma.add_argument('--end-date', default='2024-01-01', help='End date for data retrieval (YYYY-MM-DD).\n(default: 2024-01-01)')
    parser_download_risma.set_defaults(func=download_risma_command)

    # --- Download S1 command ---
    parser_download_s1 = subparsers.add_parser('download-s1', help='Downloads Sentinel-1 data via GEE.', parents=[parent_parser])
    parser_download_s1.add_argument('--stations', nargs='+', default=risma_stations, help=f"List of station IDs to process.\n(default: all stations)")
    parser_download_s1.add_argument('--buffer-distance', type=int, default=15, help='Buffer distance for S1 data (meters).\n(default: 15)')
    parser_download_s1.add_argument('--start-date', default='2015-01-01', help='Start date for data retrieval (YYYY-MM-DD).\n(default: 2015-01-01)')
    parser_download_s1.add_argument('--end-date', default='2024-01-01', help='End date for data retrieval (YYYY-MM-DD).\n(default: 2024-01-01)')
    parser_download_s1.add_argument('--roi-asset-id', required=True, help='(Required) GEE asset ID for the Region of Interest.')
    parser_download_s1.add_argument('--gee-project-id', required=True, help='(Required) Google Earth Engine project ID for S1 data download.')
    parser_download_s1.set_defaults(func=download_s1_command)

    # --- Phenology command ---
    parser_phenology = subparsers.add_parser('phenology', help='Runs the phenology model.', parents=[parent_parser])
    parser_phenology.set_defaults(func=phenology_command)

    # --- Inversion command ---
    parser_inversion = subparsers.add_parser('inversion', help='Runs the inversion algorithm.', parents=[parent_parser])
    parser_inversion.add_argument('--fghz', type=float, default=5.4, help='Frequency in GHz for inversion.\n(default: 5.4)')
    parser_inversion.add_argument('--models', default=default_models_str, help=f'RT models in JSON format.\n(default: \'{default_models_str}\')')
    parser_inversion.add_argument('--acftype', default='exp', help='ACF type for AIEM model.\n(default: exp)')
    parser_inversion.set_defaults(func=inversion_command)

    # --- Modeling command ---
    parser_modeling = subparsers.add_parser('modeling', help='Trains ensemble models and optionally deploys them to GEE.', parents=[parent_parser])
    parser_modeling.add_argument('--features', nargs='+', default=['SSM', 'vvs', 's'], help="Feature list for ML models.\n(default: 'SSM' 'vvs' 's')")
    parser_modeling.add_argument('--gee-project-id', help='Google Earth Engine project ID for model deployment.\nIf not provided, models are not deployed.')
    parser_modeling.set_defaults(func=modeling_command)

    # --- Run command (full workflow) ---
    parser_run = subparsers.add_parser(
        'run',
        help='Runs the complete workflow from download to modeling.',
        description='Runs the complete workflow from download to modeling.',
        parents=[parent_parser])
    parser_run.add_argument('--stations', nargs='+', default=risma_stations, help=f"List of station IDs to process.\n(default: all stations)")
    parser_run.add_argument('--buffer-distance', type=int, default=15, help='Buffer distance for S1 data (meters).\n(default: 15)') 
    parser_run.add_argument('--start-date', default='2015-01-01', help='Start date for data retrieval (YYYY-MM-DD).\n(default: 2015-01-01)')
    parser_run.add_argument('--end-date', default='2024-01-01', help='End date for data retrieval (YYYY-MM-DD).\n(default: 2024-01-01)')
    parser_run.add_argument('--roi-asset-id', required=True, help='(Required) GEE asset ID for the Region of Interest.')
    parser_run.add_argument('--fghz', type=float, default=5.4, help='Frequency in GHz for inversion.\n(default: 5.4)')
    parser_run.add_argument('--models', default=default_models_str, help=f'RT models in JSON format.\n(default: \'{default_models_str}\')')
    parser_run.add_argument('--acftype', default='exp', help='ACF type for AIEM model.\n(default: exp)')
    parser_run.add_argument('--features', nargs='+', default=['SSM', 'vvs', 's'], help="Feature list for ML models.\n(default: 'SSM' 'vvs' 's')")
    parser_run.add_argument('--gee-project-id', required=True, help='(Required) Google Earth Engine project ID.')
    parser_run.set_defaults(func=run_full_workflow)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
