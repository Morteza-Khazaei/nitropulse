import argparse
import sys
from .workspace import get_workspace_dir

def main():
    """Main entry point for the nitropulse CLI."""
    parser = argparse.ArgumentParser(
        prog="nitropulse",
        description="A precision tool for mapping nitrous oxide (N₂O) emission pulses in agricultural landscapes."
    )

    parser.add_argument(
        "-w", "--workspace",
        help="Path to the workspace directory. Defaults to ~/.nitropulse/workspace",
        default=None
    )

    # If no arguments are provided, print help and exit.
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Get the workspace directory, which also creates it if it doesn't exist.
    workspace = get_workspace_dir(args.workspace)

    if workspace:
        print(f"Successfully initialized workspace: {workspace}")