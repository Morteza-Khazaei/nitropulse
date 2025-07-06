import os
import click
import json

from inversion.risma import RismaData
from inversion.radar import S1Data
from inversion.pheno import BBCH
from inversion.inverse import Inverse


@click.group()
def cli():
    """Inversion Data Processing CLI."""
    pass


@cli.command()
@click.option('--workspace-dir', default='./assets', show_default=True, help='Workspace directory.')
@click.option('--auto-download', is_flag=True, show_default=True, help='Auto download data from Google Drive.')
@click.option('--stations', multiple=True, default=['MB1', 'MB2', 'MB3', 'MB4', 'MB5', 'MB6', 'MB7', 'MB8', 'MB9', 'MB10', 'MB11', 'MB12', 'MB13', 'MB14', 'MB15'], show_default=True, help='List of station IDs to process.')
@click.option('--buffer-distance', default=15, show_default=True, help='Buffer distance for Sentinel-1 data in meters.')
@click.option('--start-date', default='2010-01-01', show_default=True, help='Start date for Sentinel-1 data (YYYY-MM-DD).')
@click.option('--end-date', default='2024-01-01', show_default=True, help='End date for Sentinel-1 data (YYYY-MM-DD).')
@click.option('--gee-project-id', default=None, help='Google Earth Engine project ID.')
@click.option('--roi-asset-id', default=None, help='Region of interest asset ID in Earth Engine.')
@click.option('--fghz', default=5.4, show_default=True, help='Frequency in GHz for inversion.')
@click.option('--models', default='{"RT_s": "PRISM1", "RT_v": "Diff"}', show_default=True, help='RT models in JSON format.')
@click.option('--acftype', default='exp', show_default=True, help='ACF type for AIEM model.')


def run(
    workspace_dir, auto_download, 
    stations, buffer_distance, start_date, end_date, gee_project_id, roi_asset_id, 
    fghz, models, acftype):
    """
    Run the full inversion workflow.
    """
    click.echo("🚀 Starting inversion workflow...")
    click.echo(f"Workspace: {workspace_dir}")
    click.echo(f"Stations: {', '.join(stations)}")
    click.echo(f"Buffer distance: {buffer_distance} m")
    click.echo(f"Date range: {start_date} to {end_date}")

    # Parse models JSON
    try:
        models_dict = json.loads(models)
    except Exception as e:
        click.echo(f"❌ Error parsing models JSON: {e}")
        return

    # Initialize data handlers
    risma = RismaData(workspace_dir)
    s1 = S1Data(workspace_dir, auto_download=auto_download)

    # Download or load data
    if auto_download:
        click.echo("⬇️  Downloading S1 data...")
        s1.download(station_ids=stations, buffer_distance=buffer_distance, start_date=start_date, end_date=end_date, roi_asset=roi_asset_id, drive_folder="GEE_Exports")
    click.echo("📖 Loading RISMA and S1 data...")

    # Phenology
    click.echo("🌱 Running phenology model...")
    bbch = BBCH(workspace_dir)
    pheno_df = bbch.run()

    # Inversion
    click.echo("🔄 Running inversion...")
    inv = Inverse(workspace_dir, fGHz=fghz, models=models_dict, acftype=acftype)
    in_df = inv.run(pheno_df)

    # Save results
    output_file = os.path.join(workspace_dir, 'outputs', 'inv_df.csv')
    click.echo(f"💾 Saving results to {output_file}.")

    click.echo("✅ Inversion workflow completed!")


if __name__ == "__main__":
    cli()