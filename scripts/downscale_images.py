import cv2 as cv
from pathlib import Path
import argparse

# get directory paths from user
parser = argparse.ArgumentParser(description='Image rescaling')
parser.add_argument('--src_dir', type=str, help="Path to directory with source images")
parser.add_argument('--dest_dir', type=str, help="Path to directory for resized images")
parser.add_argument('--output_shape', type=tuple[int, int], help="Size of resized images, not including channels (default is 1024x1024)")
args = parser.parse_args()


def resize_image(src_dir, dest_dir, out_shape=(1024, 1024)):
    """Resize all images in a source directory to a given shape using 4x4 Lanczos interpolation"""

    # get list image paths
    try:
        img_paths = [path for path in Path(src_dir).iterdir()]
    except:
        raise KeyError(f"Could not find {src_dir}")

    # create the destination directory
    dest_dir = Path(dest_dir)
    try:
        dest_dir.mkdir()
    except:
        raise KeyError(f'{str(dest_dir)} already exists!')
    
    # resize images
    for path in img_paths:
        img = cv.imread(path)
        img_rs = cv.resize(img, dsize=out_shape, interpolation=cv.INTER_LANCZOS4)
        save_path = dest_dir / path.name
        cv.imwrite(save_path, img_rs)


def main():
    if args.output_shape is None:
        resize_image(args.src_dir, args.dest_dir)
    else:
        resize_image(args.src_dir, args.dest_dir, out_shape=args.output_shape)


if __name__ == '__main__':
    main()
    