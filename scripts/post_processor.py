import cv2 as cv
import numpy as np
import argparse
from skimage.segmentation import clear_border
from pathlib import Path
import math
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Literal, Union
from skimage.feature import blob_log
from sklearn.cluster import DBSCAN
from skimage.segmentation import clear_border
import pandas as pd
from scipy.spatial.distance import cityblock

ROOT = Path(__file__).parent.parent
results_dir = ROOT / ('post_processing_results' + datetime.now().strftime('_(%Y-%m-%d)_(%H-%M-%S)'))

parser = argparse.ArgumentParser(description='Segmentation Images Post-Processing')

parser.add_argument(
    '--algorithm', 
    type=str,
    choices=['CHAC', 'BubbleFinder', 'BBox'], 
    help="Type of post-processing algorithm to apply to segmentations (CHAC, BubbleFinder, BBox)"
)
parser.add_argument(
    '--seg_dir', 
    type=str, 
    help="Path to directory with input segmentation images relative to /path/to/unet-compare/"
)
parser.add_argument(
    '--img_scale', 
    type=float, 
    help="Ratio describing the number of nanometers per pixel, assuming every image in `in_dir` has the same magnification"
)
parser.add_argument(
    '--histogram_bins',
    type=float,
    nargs='+',
    help="Three space-delimited linspace-like values `start stop num` describing the bins used for computing defect size histograms"
)
parser.add_argument(
    '--tem_dir', 
    type=str,
    nargs='?',
    default=None,
    help="(Optional) Path to directory with original TEM images to be overlayed onto relative to /path/to/unet-compare/"
)
parser.add_argument(
    '--ann_dir', 
    type=str,
    nargs='?',
    default=None,
    help="(Optional) Path to directory with ground truth annotation images, if they exist, for additional ML model performance evaluations relative to /path/to/unet-compare/"
)
parser.add_argument(
    '--results_dir', 
    type=str,
    nargs='?',
    default=results_dir,
    help="(Optional) Override default path to directory for saving results relative to /path/to/unet-compare/"
)

args = parser.parse_args()

# add ROOT to results_dir if it is provided (and implicitly redefine it as a Path object)
if type(args.results_dir) == str:
    results_dir = ROOT / args.results_dir


def chac(seg_img: np.ndarray, back_img: np.ndarray, true_points: None | list, GT_exists = True) -> Union[
    tuple[int, list[float], np.ndarray, np.ndarray, list[np.ndarray]],
    tuple[int, list[float], np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[int, list[float], np.ndarray]
]:    
    # compute and visualize IOU if ground truth is available
    if GT_exists:     
        # if we are initializing ground truth (true_points is None), save the accepted polygon points, otherwise draw them for visualization
        if true_points is None:
            true_points_saved = []
        else:
            # initialize blank image for IOU operations
            iou_img_calc = np.zeros(list(seg_img.shape)+[3], dtype='uint8')
            iou_vis = iou_img_calc.copy()
            for p in true_points:
                cv.fillPoly(iou_vis, p, (0, 255, 0))
    
    # threshold and invert segmentation image
    _, seg_img = cv.threshold(seg_img, 128, 255, cv.THRESH_BINARY)
    seg_img = np.invert(seg_img)

    # further processing (morphological operations)
    k = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    seg_img = cv.morphologyEx(seg_img, cv.MORPH_OPEN, k)
    seg_img = clear_border(seg_img, buffer_size=1)

    # find contours
    contours, _ = cv.findContours(seg_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # count number of convex contours and accumulate grain diameters
    num_grains, diameters = 0, []
    for c in contours:
        # using approximate shape to replace ordinary shape
        perimeter = cv.arcLength(c, True)
        approximatedShape = cv.approxPolyDP(c, 0.02 * perimeter, True)

        # accept if convex
        ret = cv.isContourConvex(approximatedShape)
        if ret  == True:
            area = cv.contourArea(c)
            points = cv.convexHull(approximatedShape)
            
            if GT_exists:
                cv.fillPoly(iou_img_calc, [points], (255, 255, 255))
                if true_points is None:
                    # currently processing annotation images
                    true_points_saved.append([points])
                else:
                    # currently processing segmentation images
                    cv.polylines(iou_vis, [points], True, (200, 0, 200), thickness=3)
                    cv.polylines(back_img, [points], True, (200,0,200), thickness=3)
            else:
                cv.polylines(back_img, [points], True, (200,0,200), thickness=3)

            # increment counter
            num_grains += 1
            # calculate equivalent diameter
            diameters.append(2*math.sqrt(area)/math.pi*args.img_scale)

    if GT_exists:
        if true_points is None:
            # currently processing annotation images
            return num_grains, diameters, iou_img_calc, true_points_saved
        else:
            # currently processing segmentation images
            return num_grains, diameters, iou_img_calc, back_img, iou_vis, 
    else:
        # no GT exists
        return num_grains, diameters, back_img
    
    
def bubble_finder(seg_img: np.ndarray, back_img: np.ndarray, true_points: None | list, GT_exists = True) -> Union[
    tuple[int, list[float], np.ndarray, np.ndarray, list[np.ndarray]],
    tuple[int, list[float], np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[int, list[float], np.ndarray]
]:
    
    # compute and visualize IOU if ground truth is available
    if GT_exists:     
        # if we are initializing ground truth (true_points is None), save the accepted points, otherwise draw them for visualization
        if true_points is None:
            true_points_saved = []
        else:
            # initialize blank image for IOU operations
            iou_img_calc = np.zeros(list(seg_img.shape)+[3], dtype='uint8')
            iou_vis = iou_img_calc.copy()
            for p in true_points:
                cv.circle(iou_vis, center=(p[0], p[1]), radius=p[2], color=(0, 255, 0), thickness=-1)

    # threshold segmentation
    _, seg_img = cv.threshold(seg_img, 128, 255, cv.THRESH_BINARY)

    # LoG (Laplacian of Gaussian) detection
    blobs_log = blob_log(seg_img, min_sigma=1, max_sigma=9, threshold=0.25*255, overlap=0.9)
    blobs_log[:, 2] = blobs_log[:, 2] * math.sqrt(2)

    # Hough Circle Transform
    cimg = cv.GaussianBlur(seg_img, (5, 5), 0, 0)
    for i in range(cimg.shape[0]):
        for j in range(cimg.shape[1]):
            if cimg[i, j] < 0.25:
                cimg[i, j] = 0
    cimg = np.uint8(cimg * 255)
    circles = cv.HoughCircles(cimg, cv.HOUGH_GRADIENT, 1, 1, param1=0.1, param2=15, minRadius=1, maxRadius=17)
    circles = circles[0] if circles is not None else np.empty((0, 3))

    # Combine LoG and Hough Circle results
    blobs_log_xyr = np.transpose(np.array([blobs_log[:, 1], blobs_log[:, 0], blobs_log[:, 2]]))
    coords_r = np.vstack((blobs_log_xyr, circles))
    coords = coords_r[:, 0:2]

    # Apply DBSCAN clustering
    db = DBSCAN(eps=5, min_samples=2, algorithm='ball_tree').fit(coords)
    labels = db.labels_

    # Append clustering labels to the result
    coords_r = np.column_stack((coords_r, labels))

    # Aggregate cluster results
    coords_s = np.empty((0, 4))
    for i in range(0, labels.max() + 1):
        cluster = coords_r[coords_r[:, 3] == i]
        mean_cluster = np.mean(cluster, axis=0, dtype=np.float64)
        coords_s = np.append(coords_s, [mean_cluster], axis=0)

    # Append outliers
    outliers = coords_r[coords_r[:, 3] == -1]
    coords_a = np.vstack((coords_s, outliers))

    # get number of bubbles
    num_bubbles = coords_a.shape[0] 

    # draw bubbles onto segmentation image
    diameters = []
    for blob in coords_a:
        x, y, r, i = [int(el) for el in blob]

        if GT_exists:
            cv.circle(iou_img_calc, center=(x, y), radius=r, color=(255, 255, 255), thickness=-1)
            if true_points is None:
                # currently processing annotation images
                true_points_saved.append([x, y, r])
            else:
                # currently processing segmentation images
                cv.circle(iou_vis, center=(x, y), radius=r, color=(200, 0, 200), thickness=2)
                cv.circle(back_img, center=(x, y), radius=r, color=(200,0,200), thickness=2)
        else:
            cv.circle(back_img, center=(x, y), radius=r, color=(200,0,200), thickness=2)

        # calculate and append diameter
        diameters.append(2*r*args.img_scale)

    if GT_exists:
        if true_points is None:
            # currently processing annotation images
            return num_bubbles, diameters, iou_img_calc, true_points_saved
        else:
            # currently processing segmentation images
            return num_bubbles, diameters, iou_img_calc, back_img, iou_vis, 
    else:
        # no GT exists
        return num_bubbles, diameters, back_img


def bbox(seg_img: np.ndarray, back_img: np.ndarray, true_points: None | list, GT_exists = True) -> Union[
    tuple[int, list[float], np.ndarray, np.ndarray, list[np.ndarray]],
    tuple[int, list[float], np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[int, list[float], np.ndarray]
]:
    
    # compute and visualize IOU if ground truth is available
    if GT_exists:     
        # if we are initializing ground truth (true_points is None), save the accepted polygon points, otherwise draw them for visualization
        if true_points is None:
            true_points_saved = []
        else:
            # initialize blank image for IOU operations
            iou_img_calc = np.zeros(list(seg_img.shape)+[3], dtype='uint8')
            iou_vis = iou_img_calc.copy()
            for p in true_points:
                cv.rectangle(iou_vis, (p[0], p[1]), (p[0]+p[2], p[1]+p[3]), (0, 255, 0), -1)

    # initialize blank image for IOU calculation
    iou_img = np.zeros(list(seg_img.shape)+[3], dtype='uint8')

    # threshold segmentation
    _, seg_img = cv.threshold(seg_img, 128, 255, cv.THRESH_BINARY) 
    
    # find countours
    contours, _ = cv.findContours(seg_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    num_dislocations = len(contours)

    # draw dislocations onto segmentation image
    lengths = []
    for c in contours:
        bbox = cv.boundingRect(c)

        if GT_exists:
            cv.rectangle(iou_img_calc, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 255, 255), -1)
            if true_points is None:
                # currently processing annotation images
                true_points_saved.append(bbox)
            else:
                # currently processing segmentation images
                cv.rectangle(iou_vis, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (200, 0, 200), 2)
                cv.rectangle(back_img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (200,0,200), 2)
        else:
            cv.rectangle(back_img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (200,0,200), 2)

        lengths.append(math.sqrt(bbox[2]**2 + bbox[3]**2)*args.img_scale)

    if GT_exists:
        if true_points is None:
            # currently processing annotation images
            return num_dislocations, lengths, iou_img_calc, true_points_saved
        else:
            # currently processing segmentation images
            return num_dislocations, lengths, iou_img_calc, back_img, iou_vis, 
    else:
        # no GT exists
        return num_dislocations, lengths, back_img


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


def main():
    """Applies post-processing algorithm and collects results (csv and images)"""
    
    # select one of the algorithms by saving a function reference
    if args.algorithm == 'CHAC':
        algorithm = chac
    elif args.algorithm == 'BubbleFinder':
        algorithm = bubble_finder
    elif args.algorithm == 'BBox':
        algorithm = bbox

    # check if input directory exists
    seg_dir: Path = ROOT / args.seg_dir
    assert seg_dir.exists(), f'Input directory {str(seg_dir)} could not be found'

    # set a background flag and check if the TEM directory exists (if necessary)
    if args.tem_dir is not None:
        background = 'tem'
        tem_dir: Path = ROOT / args.tem_dir
        assert tem_dir.exists(), f'Input directory {str(tem_dir)} could not be found'
        tem_fps = [fp for fp in tem_dir.iterdir() if fp.is_file()]
    else:
        tem_dir = None
        background = 'blank'

    # set an annotation flag and check if the annotation directory exists (if necessary)
    if args.ann_dir is not None:
        GT_exists = True
        ann_dir: Path = ROOT / args.ann_dir
        assert ann_dir.exists(), f'Input directory {str(ann_dir)} could not be found'
        ann_fps = [fp for fp in ann_dir.iterdir() if fp.is_file()]
    else:
        ann_dir = None
        GT_exists = False

    # create results directory
    assert not results_dir.exists(), f'Results directory {results_dir.as_posix()} already exists!'
    results_dir.mkdir()

    # create directories for visualizations
    results_back_vis_dir = results_dir / 'detected_defects'
    results_back_vis_dir.mkdir()
    if GT_exists:
        results_iou_vis_dir = results_dir / 'GT_comparison'
        results_iou_vis_dir.mkdir()

    # get input segmentation file paths
    in_fps = [fp for fp in seg_dir.iterdir() if fp.is_file() and fp.suffix == '.png']

    # initialize counters and containers
    tot_num_defects = 0
    all_defect_sizes = []
    if GT_exists:
        tot_num_defects_true = 0
        all_defect_sizes_true = []
        iou_values = []

    # loop through segmentation images
    for fp in in_fps:
        # load segmentation image
        seg_img = load_image(fp, flag=cv.IMREAD_GRAYSCALE)

        # load background image
        if background == 'blank':
            back_img = np.zeros(list(seg_img.shape)+[3], dtype='uint8')
        elif background == 'tem':
            tem_path = get_matching_filepath(fp, tem_fps)
            # make sure the TEM image shape is consistent with the segmentation image shape
            back_img = load_image(tem_path, flag=cv.IMREAD_COLOR)
            assert back_img.shape[:2] == seg_img.shape, f'Expected segmentation and TEM images to have the same shape, \
                but got a TEM image at {tem_path} with shape {back_img.shape[:2]} and a segmentation image at {fp} with shape {seg_img.shape}'

        # if an annotation exists, process it and then process the segmentation image
        if GT_exists:
            ann_path = get_matching_filepath(fp, ann_fps)
            ann_img = load_image(ann_path, cv.IMREAD_GRAYSCALE)
            num_defects_true, defect_sizes_true, true_points, iou_img_true = algorithm(ann_img, None, None, GT_exists=GT_exists)

            tot_num_defects_true += num_defects_true
            for s in defect_sizes_true:
                defect_sizes_true.append([fp.stem, s])

            num_defects, defect_sizes, iou_img, back_vis, iou_vis = algorithm(seg_img, back_img, true_points, GT_exists=GT_exists)
            
            tot_num_defects += num_defects
            for s in defect_sizes:
                all_defect_sizes.append([fp.stem, s])

            # calculate and save IOU
            intersection = np.multiply(iou_img, iou_img_true)
            union = np.add(iou_img, iou_img_true) - intersection
            iou = np.sum(intersection)/np.sum(union)
            for i in iou:
                iou_values.append([fp.stem, i])

            # save visualizations
            cv.imwrite(results_back_vis_dir / fp.name, back_vis)
            cv.imwrite(results_iou_vis_dir / fp.name, iou_vis)
            
        else:
            num_defects, defect_sizes, back_vis = algorithm(seg_img, back_img, None, GT_exists=GT_exists)
            
            tot_num_defects += num_defects
            for s in defect_sizes:
                all_defect_sizes.update({fp.stem: s})

            # save visualization
            cv.imwrite(results_back_vis_dir / fp.name, back_vis)

    # save data to csv files
    pd.DataFrame(all_defect_sizes).to_csv(results_dir / 'defect_sizes.csv', index=True, sep='\t')
    if GT_exists:
        pd.DataFrame(all_defect_sizes_true).to_csv(results_dir / 'GT_defect_sizes.csv', index=True, sep='\t')
        pd.DataFrame(iou_values).to_csv(results_dir / 'iou.csv', index=True, sep='\t')

    # generate list of defect sizes from dictionaries
    all_defect_sizes_just_vals = []
    for key, val in all_defect_sizes:
        all_defect_sizes_just_vals.append(val)
    
    if GT_exists:
        all_defect_sizes_true_just_vals = []
        for key, val in all_defect_sizes_true:
            all_defect_sizes_true_just_vals.append(val)

        iou_values_just_vals = []
        for key, val in iou_values:
            iou_values_just_vals.append(val)

    # compute and save defect size histogram
    bins = np.linspace(args.bins[0], args.bins[1], args.bins[2])
    seg_histogram, seg_bins = np.histogram(all_defect_sizes_just_vals, bins=bins, density=False)
    bar_width = max(bins)/len(bins-1)*0.2
    fig, ax = plt.subplots()
    if GT_exists:
        true_histogram, true_bins = np.histogram(all_defect_sizes_true_just_vals, bins=bins, density=False)
        l1_distance = cityblock(seg_histogram, true_histogram)
        bar_offset = bar_width*0.8
        ax.bar(true_bins[:-1]-bar_offset, true_histogram, width=bar_width, color='green', label='True', edgecolor='black')
        ax.bar(seg_bins[:-1]+bar_offset, seg_bins, width=bar_width, color='purple', label='Prediction', edgecolor='black')   
    else:
        ax.bar(seg_bins[:-1], seg_bins, width=bar_width, color='purple', label='Prediction', edgecolor='black')   
    ax.legend()
    ax.set_xlabel('Defect Size [nm]')
    ax.set_ylabel('Counts')
    ax.set_title(f'Defect size histogram')
    fig.savefig(results_dir / 'defect_size_histogram.png')
    plt.close(fig)
    
    # save info to a text file
    with open(results_dir / 'info.out', 'rw') as f:
        f.write(f'Selected algorithm: {args.algorithm}\n\n')
        f.write(f'Segmentation directory path: {seg_dir}\n')
        f.write(f'TEM directory path: {tem_dir}\n')
        f.write(f'Annotation directory path: {ann_dir}\n\n')
        f.write(f'Number of detected defects: {tot_num_defects}\n')
        f.write(f'Defect size mean (nm): {np.mean(all_defect_sizes_just_vals)}\n')
        f.write(f'Defect size standard deviation (nm): {np.std(all_defect_sizes_just_vals)}\n\n')
        if GT_exists:
            f.write(f'GT number of detected defects (GT): {tot_num_defects_true}\n')
            f.write(f'GT defect size mean (nm): {np.mean(all_defect_sizes_true_just_vals)}\n')
            f.write(f'GT defect size standard deviation (nm): {np.std(all_defect_sizes_true_just_vals)}\n\n')
            f.write(f'IOU mean: {np.mean(iou_values_just_vals)}\n')
            f.write(f'IOU standard deviation: {np.std(iou_values_just_vals)}\n\n')
            f.write(f'Defect size histogram L1-distance: {l1_distance}')


main()