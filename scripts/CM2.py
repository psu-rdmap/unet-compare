"""
Creates confusion matrix color mappings (CM2)
"""

import cv2 as cv
import numpy as np
from pathlib import Path
import argparse, glob
from datetime import datetime

ROOT = Path(__file__).parent.parent
results_path = ROOT / ('CM2' + datetime.now().strftime('_(%Y-%m-%d)_(%H-%M-%S)'))

parser = argparse.ArgumentParser(description='CM2 Algorithm')

parser.add_argument(
    '--img_dir', 
    type=str,
    default=None, 
    help="Path to directory with training images corresponding to files in `pred_dir` relative to /path/to/unet-compare/"
)
parser.add_argument(
    '--ann_dir', 
    type=str, 
    help="Path to directory with ground truth annotation images corresponding to files in `pred_dir` relative to /path/to/unet-compare/"
)
parser.add_argument(
    '--pred_dir', 
    type=str, 
    help="Path to directory with model predictions relative to /path/to/unet-compare/"
)
parser.add_argument(
    '--results_dir', 
    type=str | Path,
    nargs='?',
    default=results_path,
    help="(Optional) Override default path to directory for saving CM2 images relative to /path/to/unet-compare/" 
)
args = parser.parse_args()

# add ROOT to results_dir if it is provided (and implicitly redefine it as a Path object)
if type(args.results_dir) == str:
    results_dir = ROOT / args.results_dir


def load_image(path: Path, flag = cv.IMREAD_GRAYSCALE | cv.IMREAD_COLOR) -> np.ndarray:
    img = cv.imread(path.as_posix(), flag)
    assert type(img) == np.ndarray, f"File at {path} could not be loaded!"
    return img
    

def get_matching_filepath(target_fn: str, potential_fps: list[Path]) -> Path:
    "Find the full filepath from a list of potential filepaths given just a filename"
    found_fp = None
    for fp in potential_fps:
        if fp.stem == target_fn:
            found_fp = fp
    
    if found_fp is None:
        raise ValueError(f'Could not find file in {potential_fps[0].parent.as_posix()} matching the file name {target_fn}')
    else:
        return found_fp
        
        
def binarize_img(img: np.ndarray) -> np.ndarray: 
    """Threshold the input image using Otsu's method and return an array of 0s and 1s"""
    _, img_threshd = cv.threshold(img.copy(), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return (img_threshd/255).astype(int)


def get_mask(ann_bin: np.ndarray, pred_bin: np.ndarray, compare_condition: tuple[int, int]) -> np.ndarray:
    """Given the compare condition (ints are 0 or 1), compare the annotation and prediction"""
    mask = np.logical_and(ann_bin==compare_condition[0], pred_bin==compare_condition[1])
    return mask.astype('uint8')


def CM2_single_image(img_path: Path, ann_path: Path, pred_path: Path, results_dir: Path):
    """Apply the CM2 algorithm to a single image, annotation, prediction pair and overlay onto the image background is None"""
    
    # load images
    img = load_image(img_path, flag=cv.IMREAD_COLOR)
    ann = load_image(ann_path, flag=cv.IMREAD_GRAYSCALE)
    pred = load_image(pred_path, flag=cv.IMREAD_GRAYSCALE)

    # check shapes
    assert pred.shape == img.shape[:2], f"Expected training image and prediction to have the same shape. \
        Got an image at {img_path} with shape {img.shape[:2]} and prediction {pred_path} with shape {pred.shape}"
    assert pred.shape == ann.shape, f"Expected annotation and prediction to have the same shape. \
        Got an annotation at {ann_path} with shape {ann.shape} and prediction {pred_path} with shape {pred.shape}"

    # binarize annotation and prediction
    ann_bin = binarize_img(ann)
    pred_bin = binarize_img(pred)

    # generate masks with pixels classified according to confusion matrix
    tp_mask = get_mask(ann_bin, pred_bin, compare_condition=(1,1))
    tn_mask = get_mask(ann_bin, pred_bin, compare_condition=(0,0))
    fp_mask = get_mask(ann_bin, pred_bin, compare_condition=(0,1))
    fn_mask = get_mask(ann_bin, pred_bin, compare_condition=(1,0))

    # define background base channel
    base_channel = np.multiply(tn_mask, img)
    
    # define color channels as base channel overlayed with masks
    b_channel = np.add(base_channel, fp_mask*255).astype('uint8')
    g_channel = np.add(base_channel, tp_mask*255).astype('uint8')
    r_channel = np.add(base_channel, fn_mask*255).astype('uint8')

    # create composite mask (opencv uses BGR) and move image channel axis to end for correct shape
    cm2 = np.array([b_channel, g_channel, r_channel])
    cm2 = np.moveaxis(cm2, 0, -1)

    # save confusion mask
    cv.imwrite((results_dir / pred_path.name).as_posix(), cm2)


def main():
    # make sure all input directories exist
    img_dir = Path(ROOT / args.img_dir)
    ann_dir = Path(ROOT / args.ann_dir)
    pred_dir = Path(ROOT / args.pred_dir)
    assert img_dir.exists(), f"tem_dir {img_dir} does not exist!"
    assert ann_dir.exists(), f"ann_dir {ann_dir} does not exist!"
    assert pred_dir.exists(), f"pred_dir {pred_dir} does not exist!"

    # make sure results_dir does not already exist, then create it
    assert not results_dir.exists(), f"results_dir {results_dir} already exists!"
    results_dir.mkdir(parents=True)
    
    # get file paths in each directory
    img_fps = [p for p in img_dir.iterdir() if p.is_file()]
    ann_fps = [p for p in ann_dir.iterdir() if p.is_file()]
    pred_fps = [p for p in pred_dir.iterdir() if p.is_file()]

    # loop through preds
    for pred_path in pred_fps:
        # find corresponding TEM and annotation images
        tem_path = get_matching_filepath(pred_path, img_fps)
        ann_path = get_matching_filepath(pred_path, ann_fps)

        # run CM2 algroithm
        CM2_single_image(tem_path, ann_path, pred_path, results_dir)


main()