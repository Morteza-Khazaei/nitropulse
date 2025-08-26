import os
import argparse
import json

from inversion.risma import RismaData
from inversion.radar import S1Data
from inversion.pheno import BBCH
from inversion.inverse import Inverse
from inversion.ensemble_rf import RegressionRF


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
    s1 = S1Data(args.workspace_dir, auto_download=True)

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
    s1.download(
        station_ids=args.stations,
        buffer_distance=args.buffer_distance,
        start_date=args.start_date,
        end_date=args.end_date,
        roi_asset=args.roi_asset_id,
        drive_folder="GEE_Exports"
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
        print(f" Phenology output file not found at {pheno_df_path}")
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


def main():
    """
    A command-line interface for running SAR backscatter inversion models.
    """
    home_dir = os.path.expanduser("~")
    default_workspace = os.path.join(home_dir, '.inversion')
    if not os.path.exists(default_workspace):
        os.makedirs(default_workspace)
        
    risma_stations = ['RISMA_MB1', 'RISMA_MB2', 'RISMA_MB3', 'RISMA_MB4', 'RISMA_MB5',
                      'RISMA_MB6', 'RISMA_MB7', 'RISMA_MB8', 'RISMA_MB9', 'RISMA_MB10',
                      'RISMA_MB11', 'RISMA_MB12', 'RISMA_MB13']

    parser = argparse.ArgumentParser(
        description='A command-line interface for running SAR backscatter inversion models.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # Download command
    download_parser = subparsers.add_parser('download', help='Download RISMA and Sentinel-1 data')
    download_parser.add_argument('--workspace-dir', default=default_workspace, help='Workspace directory')
    download_parser.add_argument('--stations', nargs='*', default=risma_stations, help='List of station IDs')
    download_parser.add_argument('--buffer-distance', type=int, default=15, help='Buffer distance for S1 data')
    download_parser.add_argument('--start-date', default='2010-01-01', help='Start date for data')
    download_parser.add_argument('--end-date', default='2024-01-01', help='End date for data')
    download_parser.add_argument('--roi-asset-id', help='ROI asset ID in Earth Engine')
    download_parser.set_defaults(func=run_download)

    # Phenology command
    phenology_parser = subparsers.add_parser('phenology', help='Run the phenology model')
    phenology_parser.add_argument('--workspace-dir', default=default_workspace, help='Workspace directory')
    phenology_parser.set_defaults(func=run_phenology)

    # Inversion command
    inversion_parser = subparsers.add_parser('inversion', help='Run the inversion algorithm')
    inversion_parser.add_argument('--workspace-dir', default=default_workspace, help='Workspace directory')
    inversion_parser.add_argument('--fghz', type=float, default=5.4, help='Frequency in GHz')
    inversion_parser.add_argument('--models', default='{"RT_s": "PRISM1", "RT_c": "Diff"}', help='RT models in JSON format')
    inversion_parser.add_argument('--acftype', default='exp', help='ACF type for AIEM model')
    inversion_parser.set_defaults(func=run_inversion)

    # Modeling command
    modeling_parser = subparsers.add_parser('modeling', help='Train ensemble models and deploy to GEE')
    modeling_parser.add_argument('--workspace-dir', default=default_workspace, help='Workspace directory')
    modeling_parser.add_argument('--features', nargs='*', default=['SSM', 'vvs', 's'], help='Feature list')
    modeling_parser.add_argument('--gee-project-id', help='Google Earth Engine project ID')
    modeling_parser.set_defaults(func=run_modeling)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()