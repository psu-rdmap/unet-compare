from pathlib import Path
import os
import argparse

parser = argparse.ArgumentParser(description='Remove large model files for export')
parser.add_argument('root_dir', type=str, help="Path to directory tree containing model files to be erased relative to /path/to/unet-compare/")
args = parser.parse_args()

root_dir = Path(args.root_dir)
assert root_dir.exists(), f"{root_dir} is not a valid directory"

for path in root_dir.rglob('*.keras'):
    os.remove(path)
