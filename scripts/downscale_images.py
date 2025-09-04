import cv2 as cv
from pathlib import Path
import argparse
import numpy as np

ROOT = Path(__file__).parent.parent

# get directory paths from user
parser = argparse.ArgumentParser(description='Image rescaling')

parser.add_argument(
    '--in_dir', 
    type=str, 
    help="Path to directory with input images relative to /path/to/unet-compare/"
)
parser.add_argument(
    '--out_dir', 
    type=str, 
    help="Path to directory for saving downscaled images relative to /path/to/unet-compare/"
)
parser.add_argument(
    '--resize_shape', 
    type=int,
    nargs='+',
    default=[1024, 1024],
    help="Shape to downscale input images to (e.g., `1024 1024`)"
)
args = parser.parse_args()

def load_image(path: Path) -> np.ndarray:
    img = cv.imread(path.as_posix(), cv.IMREAD_ANYCOLOR)
    assert type(img) == np.ndarray, f"File at {path} could not be loaded!"
    return img


def main():
    """Resize all images in an input directory to a given shape using 4x4 Lanczos interpolation"""

    # check in_dir
    in_dir = Path(ROOT / args.in_dir)
    assert in_dir.exists(), f"in_dir {in_dir} does not exist!"

    # check uniqueness of out_dir and create it
    out_dir = Path(ROOT / args.out_dir)
    assert in_dir.as_posix() != out_dir.as_posix(), f"Expected in_dir and out_dir must be different. Got {in_dir} and {out_dir}, respectively"
    assert not out_dir.exists(), f"out_dir already exists!"
    out_dir.mkdir(parents=True)

    # run through files and resize them
    in_fps = [p for p in in_dir.iterdir() if p.is_file()]
    for fp in in_fps:
        img = load_image(fp)

        # make sure we are downscaling
        assert (img.shape[0] > args.resize_shape[0]) and (img.shape[1] > args.resize_shape[1]), f"Expected input image {fp} to be larger than resize shape in both dimensions. \
            Got image shape {img.shape[:2]} and resized shape {args.resize_shape}"
        
        # resize
        resize_shape = args.resize_shape + [img.shape[:-1]]
        img_rs = cv.resize(img, dsize=args.resize_shape, interpolation=cv.INTER_LANCZOS4)

        # save new image
        cv.imwrite((out_dir / fp.name).as_posix(), img_rs)

main()