import os
import argparse
import json

from inversion.risma import RismaData
from inversion.radar import S1Data
from inversion.pheno import BBCH
from inversion.inverse import Inverse
from inversion.ensemble_rf import RegressionRF


def main():
    """
    A command-line interface for running SAR backscatter inversion models.

    This tool orchestrates the data preparation, phenology modeling,
    and final inversion steps to estimate soil and vegetation parameters
    from Sentinel-1 radar data based on the ground-based soil surface parameters 
    measurements from RISMA networks.
    """
    
    risma_stations = ['RISMA_MB1', 'RISMA_MB2', 'RISMA_MB3', 'RISMA_MB4', 'RISMA_MB5', 
                      'RISMA_MB6', 'RISMA_MB7', 'RISMA_MB8', 'RISMA_MB9', 'RISMA_MB10', 
                      'RISMA_MB11', 'RISMA_MB12', 'RISMA_MB13']
    
    parser = argparse.ArgumentParser(
        description='Run SAR backscatter inversion models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
    python inversion.py --workspace-dir ./data --auto-download \\
                       --stations MB1 MB2 --gee-project-id your-gcp-project
        """
    )
    
    # Workspace and general options
    parser.add_argument('--workspace-dir', 
                       default='./assets', 
                       help='Workspace directory (default: ./assets)')
    
    parser.add_argument('--auto-download', 
                       action='store_true',
                       help='Auto download Sentinel-1 data via Google Earth Engine')
    
    # Station selection - simplified multiple argument handling
    parser.add_argument('--stations', 
                       nargs='*',  # Accept zero or more arguments
                       default=risma_stations,
                       help='List of station IDs to process (space-separated)')
    
    # Data parameters
    parser.add_argument('--buffer-distance', 
                       type=int,
                       default=15,
                       help='Buffer distance for Sentinel-1 data in meters (default: 15)')
    
    parser.add_argument('--start-date', 
                       default='2010-01-01',
                       help='Start date for Sentinel-1 data (YYYY-MM-DD) (default: 2010-01-01)')
    
    parser.add_argument('--end-date', 
                       default='2024-01-01',
                       help='End date for Sentinel-1 data (YYYY-MM-DD) (default: 2024-01-01)')
    
    # GEE options
    parser.add_argument('--gee-project-id', 
                       help='Google Earth Engine project ID')
    
    parser.add_argument('--roi-asset-id', 
                       help='Region of interest asset ID in Earth Engine')
    
    # Inversion parameters
    parser.add_argument('--fghz', 
                       type=float,
                       default=5.4,
                       help='Frequency in GHz for inversion (default: 5.4)')
    
    parser.add_argument('--models', 
                       default='{"RT_s": "PRISM1", "RT_c": "Diff"}',
                       help='RT models in JSON format (default: {"RT_s": "PRISM1", "RT_c": "Diff"})')
    
    parser.add_argument('--acftype', 
                       default='exp',
                       help='ACF type for AIEM model (default: exp)')
    
    # Features - simplified list handling
    parser.add_argument('--features', 
                       nargs='*',
                       default=['SSM', 'vvs', 's'],
                       help='Feature list based on RT_s model outputs (space-separated)')
    
    args = parser.parse_args()
    
    # Run the inversion workflow
    run_inversion(args)


def run_inversion(args):
    """
    Run the full inversion workflow from data download to result generation.

    This function performs the following steps:
    1. Initializes data handlers for RISMA (ground-truth) and Sentinel-1 data.
    2. Optionally downloads Sentinel-1 data from GEE for specified stations.
    3. Runs a phenology model (BBCH) to determine crop growth stages.
    4. Executes the core inversion algorithm using the prepared data.
    5. Saves the final inverted parameters to a CSV file.
    """
    
    print("🚀 Starting inversion workflow...")
    print(f"Workspace: {args.workspace_dir}")
    print(f"Stations: {', '.join(args.stations)}")
    print(f"Buffer distance: {args.buffer_distance} m")
    print(f"Date range: {args.start_date} to {args.end_date}")

    # Parse models JSON
    try:
        models_dict = json.loads(args.models)
    except Exception as e:
        print(f"❌ Error parsing models JSON: {e}")
        return

    # Initialize data handlers
    risma = RismaData(args.workspace_dir)
    s1 = S1Data(args.workspace_dir, auto_download=args.auto_download)

    # Download or load data
    if args.auto_download:
        print("⬇️  Downloading RISMA data...")
        risma.download_risma_data(
            out_dir='../assets/inputs/RISMA_CSV_SSM_SST_AirT_2015_2023_new', 
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
    
    print("📖 Loading RISMA and S1 data...")

    # Phenology
    print("🌱 Running phenology model...")
    bbch = BBCH(args.workspace_dir)
    pheno_df = bbch.run()

    # Save phenology results
    output_file = os.path.join(args.workspace_dir, 'outputs', 'pheno_df.csv')
    print(f"💾 Saving phenology results to {output_file}.")
    pheno_df.to_csv(output_file, index=False)

    # Inversion
    print("🔄 Running inversion...")
    inv = Inverse(args.workspace_dir, fGHz=args.fghz, models=models_dict, acftype=args.acftype)
    inv_df = inv.run(pheno_df)

    # Save inversion results
    output_file = os.path.join(args.workspace_dir, 'outputs', 'inv_df.csv')
    print(f"💾 Saving inversion results to {output_file}.")
    inv_df.to_csv(output_file, index=False)
    print("✅ Inversion workflow completed!")

    print()

    # Train the ensemble models and send them to GEE
    print('🔄 Running ensemble training...')
    ensrf = RegressionRF(args.workspace_dir, inv_df)
    rf_models = ensrf.run(vars=list(args.features))
    print('✅ Ensemble training completed!')
    
    if args.gee_project_id:
        print('🔄 Uploading ensemble models to GEE...')
        passed = ensrf.upload_rf_to_gee(rf_models, args.gee_project_id, save_dectree=False)
        print('✅ Ensemble models uploaded to GEE!')
    else:
        print('⚠️  No GEE project ID provided, skipping upload to GEE.')


if __name__ == "__main__":
    main()