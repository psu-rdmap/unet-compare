"""
Creates confusion matrix color mappings (CM2)
"""

import cv2 as cv
import numpy as np
from pathlib import Path
import argparse, glob

ROOT = Path.cwd()

parser = argparse.ArgumentParser(description='CM2 Algorithm')
parser.add_argument(
    'image_dir', 
    type=str, 
    help="Path to directory with images relative to /path/to/unet-compare/"
)
parser.add_argument(
    'annotation_dir', 
    type=str, 
    help="Path to directory with annotations relative to /path/to/unet-compare/"
)
parser.add_argument(
    'prediction_dir', 
    type=str, 
    help="Path to directory with predictions relative to /path/to/unet-compare/"
)
parser.add_argument(
    'save_dir', 
    type=str, 
    help="Path to directory to save color mappings relative to /path/to/unet-compare/"
)
args = parser.parse_args()


def load_img(img_path: Path):
    """Load an image from the path as a grayscale numpy array"""
    img = cv.imread(img_path.as_posix(), cv.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Image at {img_path} could not be loaded.')
    else:
        return img
    

def binarize_img(img: np.ndarray): 
    """Threshold the input image using Otsu's method and return an array of 0s and 1s"""
    _, img_threshd = cv.threshold(img.copy(), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return (img_threshd/255).astype(int)


def get_mask(ann_bin: np.ndarray, pred_bin: np.ndarray, compare_condition: tuple[int, int]):
    """Given the compare condition (ints are 0 or 1), compare the annotation and prediction"""
    mask = np.logical_and(ann_bin==compare_condition[0], pred_bin==compare_condition[1])
    return mask.astype('uint8')


def CM2_single_image(img_path: Path, ann_path: Path, pred_path: Path, save_dir: Path):
    """Apply the CM2 algorithm to a single image, annotation, prediction pair and overlay onto the image background is None"""
    
    # load images
    img = load_img(img_path)
    ann = load_img(ann_path)
    pred = load_img(pred_path)

    # check shapes
    assert pred.shape == img.shape, f"Image {str(img_path)} has shape {img.shape} and prediction {str(pred_path)} has shape {pred.shape}. \
    They must have the same shape"
    assert pred.shape == ann.shape, f"Annotation {str(ann_path)} has shape {ann.shape} and prediction {str(pred_path)} has shape {pred.shape}. \
    They must have the same shape"

    # binarize annotation and prediction
    ann_bin = binarize_img(ann)
    pred_bin = binarize_img(pred)

    # generate masks with pixels classified according to confusion matrix
    tp_mask = get_mask(ann_bin, pred_bin, compare_condition=(1,1))
    tn_mask = get_mask(ann_bin, pred_bin, compare_condition=(0,0))
    fp_mask = get_mask(ann_bin, pred_bin, compare_condition=(0,1))
    fn_mask = get_mask(ann_bin, pred_bin, compare_condition=(1,0))

    # define background base channel
    base_channel = base_channel = np.multiply(tn_mask, img)
    
    # define color channels as base channel overlayed with masks
    b_channel = np.add(base_channel, fp_mask*255).astype('uint8')
    g_channel = np.add(base_channel, tp_mask*255).astype('uint8')
    r_channel = np.add(base_channel, fn_mask*255).astype('uint8')

    # create composite mask (opencv uses BGR) and move image channel axis to end for correct shape
    cm2 = np.array([b_channel, g_channel, r_channel])
    cm2 = np.moveaxis(cm2, 0, -1)

    # save confusion mask
    cv.imwrite((save_dir / pred_path.name).as_posix(), cm2)


def main():
    # create Path objects from input directories
    img_dir = ROOT / args.image_dir
    ann_dir = ROOT / args.annotation_dir
    pred_dir = ROOT / args.prediction_dir
    save_dir = ROOT / args.save_dir

    # create save_dir
    if not save_dir.exists():
        save_dir.mkdir()

    # loop through preds
    for pred_path in pred_dir.iterdir():
        if pred_path.is_file():
            # determine corresponding image and annotation paths
            img_path = img_dir / pred_path.stem
            img_path = Path(glob.glob(img_path.as_posix() + '*')[0])
            assert img_path.exists(), f"Image file {img_path} does not exist"
            
            ann_path = ann_dir / pred_path.stem
            ann_path = Path(glob.glob(ann_path.as_posix() + '*')[0])
            assert ann_path.exists(), f"Annotation file {ann_path} does not exist"

            # run CM2 algroithm
            CM2_single_image(img_path, ann_path, pred_path, save_dir)


main()