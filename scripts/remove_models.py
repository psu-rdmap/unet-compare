from pathlib import Path
import os
import argparse

parser = argparse.ArgumentParser(description='Remove large model files for export')
parser.add_argument('root_dir', type=str, help="Path to the start of a directory tree where `.keras` files will be removed")
args = parser.parse_args()

root_dir = Path(args.root_dir)
assert root_dir.exists(), f"{root_dir} is not a valid directory"

for path in root_dir.rglob('*.keras'):
    os.remove(path)
