"""
CHAC algorithm was developed by Xinyuan Xu
https://doi.org/10.1016/j.jnucmat.2023.154813
"""

import cv2 as cv
import numpy as np
import argparse
from skimage.segmentation import clear_border
from pathlib import Path
from warnings import warn
import glob
import matplotlib.pyplot as plt

ROOT = Path.cwd()

parser = argparse.ArgumentParser(description='CHAC Algorithm')
parser.add_argument(
    'src_dir', 
    type=str, 
    help="Path to directory with source images relative to /path/to/unet-compare/"
)
parser.add_argument(
    'save_dir', 
    type=str, 
    help="Path to directory for saved grain images relative to /path/to/unet-compare/"
)
parser.add_argument(
    '--background_dir', 
    type=str, 
    help="Optional path to directory with background images for overlaying grains relative to /path/to/unet-compare/. " \
    "Background images should have the same filename as source image"
)
args = parser.parse_args()

def CHAC_single_image(path: Path, save_dir: Path, background_dir: Path):
    """Loads a grain boundary segmentation, detects convex grains, overlays it on a background, and saves the overlay"""
    
    # load image
    image = cv.imread(path.as_posix(), cv.IMREAD_COLOR)
    if image is None:
        warn(f"Image at {str(path)} could not be loaded by OpenCV. Skipping this file")
        return

    # check dest_dir
    if not save_dir.exists():
        save_dir.mkdir()
        
    if background_dir is None:
        # set background as white if not supplied
        background = np.zeros(image.shape, dtype=np.uint8)
        background.fill(255)
    else:
        # background is supplied in background_dir with the same name
        assert background_dir.exists(), f'Background directory {str(background_dir)} could not be found'
        background_path = glob.glob((background_dir / path.stem).as_posix() + '.*')[0]
        background = cv.imread(background_path)
        assert background is not None, f"Image at {background_path} could not be loaded by OpenCV"
        assert background.shape[0:1] == image.shape[0:1], f"Source image {str(path)} has shape {image.shape[:2]}, but background image {background_path} \
        has shape {background.shape[:2]}. They must be the same shape"
        
    # threshold segmentation
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, image_bin = cv.threshold(image_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    image_bin = np.invert(image_bin)

    # get structring elements
    k = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # open operation
    image_bin = cv.morphologyEx(image_bin, cv.MORPH_OPEN, k)
    # remove edge touching grains
    image_bin = clear_border(image_bin, buffer_size=1)

    # find contours
    contours, _ = cv.findContours(image_bin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # count convex contours and total contours
    convex_contours = 0
    total_contours = 0

    # determine grain size distribution
    areas = []

    for c in range(len(contours)):
        # using approximate shape to replace ordinary shape
        perimeter = cv.arcLength(contours[c], True)
        approximatedShape = cv.approxPolyDP(contours[c], 0.01 * perimeter, True)
        
        # convex or not
        ret = cv.isContourConvex(approximatedShape)
        # convex detect
        if ret  == True:
            points = cv.convexHull(approximatedShape)
            total = len(points)
            for i in range(len(points)):
                x1, y1 = points[i % total][0]
                x2, y2 = points[(i+1) % total][0]
                cv.line(background, (x1, y1), (x2, y2), (200, 0, 200), 3, 8, 0)
            convex_contours += 1
            total_contours += 1
            areas.append(cv.contourArea(contours[c]))
            
        else:
            total_contours += 1

    # calculate diameter histogram
    scale = 100 / 110 # nm/px
    diameters = 2*np.sqrt(np.array(areas))/np.pi*scale

    # save image
    cv.imwrite((save_dir / path.name).as_posix(), background)

    return diameters


def main():
    """Applies CHAC algorithm to a directory with images"""
        
    src_dir = ROOT / args.src_dir
    assert src_dir.exists(), f'Source directory {str(src_dir)} could not be found'

    save_dir = ROOT / args.save_dir
    if args.background_dir:
        background_dir = ROOT / args.background_dir
    else:
        background_dir = None
    
    diameters = np.array([])
    for path in src_dir.iterdir():
        if path.is_file():
            d = CHAC_single_image(path, save_dir, background_dir)
            diameters = np.concatenate([diameters, d])

    # compute grain size distribution
    hist, bin_edges = np.histogram(diameters, bins=7, density=False)
    
    plt.bar(bin_edges[:-1], hist, width=7)
    plt.xlabel('Effective Grain Diameter [nm]')
    plt.ylabel('Grain Counts')
    plt.title('Grain Size Distribution')

    plt.savefig((save_dir / 'grain_size_distribution.png').as_posix(), bbox_inches='tight')

    print("Grain diameter mean:", round(np.average(diameters), 3), 'nm')
    print("Grain diamter standard deviation:", round(np.std(diameters), 3), 'nm')


main()