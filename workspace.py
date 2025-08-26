import pathlib
from typing import Optional

def get_workspace_dir(workspace_arg: Optional[str] = None) -> Optional[pathlib.Path]:
    """
    Determines the workspace directory path.

    If a workspace path is provided via `workspace_arg`, it is resolved and used.
    Otherwise, it defaults to a '.nitropulse/workspace' directory within the
    user's home directory.

    The directory is created if it doesn't exist.

    Args:
        workspace_arg: The optional path to the workspace directory provided by the user.

    Returns:
        A Path object to the workspace directory, or None if creation fails.
    """
    if workspace_arg:
        workspace_path = pathlib.Path(workspace_arg).resolve()
    else:
        home_dir = pathlib.Path.home()
        # Use a hidden directory for application data, named after the package.
        workspace_path = home_dir / ".nitropulse" / "workspace"

    try:
        workspace_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create workspace directory at {workspace_path}: {e}")
        return None

    return workspace_path