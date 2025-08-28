import os
import shutil
from setuptools import setup

# This script provides a custom build step to handle a non-standard project layout
# where configuration files are stored outside the 'src' directory. The standard
# and recommended practice for a 'src' layout is to place all package data
# inside the 'src/<package_name>' directory.

CONFIG_SOURCE_DIR = 'assets/config'
CONFIG_DEST_DIR = 'src/nitropulse/config'

# This pre-build step copies the config files into the source tree.
# This allows 'setuptools' to find and package them correctly, and also makes
# them available for 'pip install -e .' (editable mode).
if os.path.isdir(CONFIG_SOURCE_DIR):
    print(f"--- Pre-build: Copying config files from '{CONFIG_SOURCE_DIR}' to '{CONFIG_DEST_DIR}' ---")
    if os.path.isdir(CONFIG_DEST_DIR):
        shutil.rmtree(CONFIG_DEST_DIR)
    shutil.copytree(CONFIG_SOURCE_DIR, CONFIG_DEST_DIR)
    print("--- Pre-build: Copy complete. ---")

# The main setup() call will read its configuration from 'pyproject.toml'.
setup()