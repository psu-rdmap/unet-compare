import cv2 as cv
from pathlib import Path
import argparse

def list_of_ints(arg) -> list[int, int]:
    return list(map(int, arg.split(',')))

# get directory paths from user
parser = argparse.ArgumentParser(description='Image rescaling')
parser.add_argument('--in_dir', type=str, help="Path to directory with input images relative to /path/to/unet-compare/")
parser.add_argument('--out_dir', type=str, help="Path to directory for saving downscaled images relative to /path/to/unet-compare/")
parser.add_argument('--out_shape', type=list_of_ints, help="Shape to downscale input images to (e.g., [1024, 1024])")
args = parser.parse_args()

def resize_image(src_dir, dest_dir, out_shape=[1024, 1024]):
    """Resize all images in a source directory to a given shape using 4x4 Lanczos interpolation"""

    # get list image paths
    try:
        img_paths = [path for path in Path(src_dir).iterdir()]
    except:
        raise KeyError(f"Could not find {src_dir}")

    # create the destination directory
    dest_dir = Path(dest_dir)
    if dest_dir.exists():
        raise KeyError(f'{str(dest_dir)} already exists!')
    else:
        dest_dir.mkdir(exist_ok=True, parents=True)  
    
    # resize images
    for path in img_paths:
        img = cv.imread(path)
        img_rs = cv.resize(img, dsize=out_shape, interpolation=cv.INTER_LANCZOS4)
        save_path = dest_dir / path.name
        cv.imwrite(save_path, img_rs)


def main():
    assert args.src_dir != args.dest_dir, 'Source and destination directories must have different names!'

    if args.output_shape is None:
        resize_image(args.src_dir, args.dest_dir)
    else:
        resize_image(args.src_dir, args.dest_dir, out_shape=args.output_shape)


if __name__ == '__main__':
    main()
    