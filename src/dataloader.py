"""
Aiden Ochoa, 4/2025, RDMAP PSU Research Group
This module handles dataset processing for training and inference
"""

from typing import Tuple, List
from pathlib import Path
import shutil, random, re
from natsort import os_sorted
from glob import glob
import cv2 as cv
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
tf.random.set_seed(3051)
random.seed(229)


def create_train_dataset(configs: dict) -> dict:
    """Creates the training dataset based on the configs"""

    data_dir: Path = configs['root_dir'] / 'data' / configs['dataset_name']
    dataset_dir: Path = configs['root_dir'] / 'dataset'

    # Step 1: remove existing directory if it exists
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    # Step 2: generate train/val filename lists
    configs['training_set'], configs['validation_set'] = split_data(configs)

    # Step 3: convert filename stems to full paths
    img_ext = next(iter({file.suffix for file in (data_dir / 'images').iterdir()}))
    ann_ext = next(iter({file.suffix for file in (data_dir / 'annotations').iterdir()}))

    train_img_paths = [data_dir / 'images' / (file + img_ext) for file in configs['training_set']]
    val_img_paths = [data_dir / 'images' / (file + img_ext) for file in configs['validation_set']]
    train_ann_paths = [data_dir / 'annotations' / (file + ann_ext) for file in configs['training_set']]
    val_ann_paths = [data_dir / 'annotations' / (file + ann_ext) for file in configs['validation_set']]

    # Step 4: populate dataset tree
    copy_files(train_img_paths, dataset_dir / 'images' / 'train')
    copy_files(val_img_paths, dataset_dir / 'images' / 'val')
    copy_files(train_ann_paths, dataset_dir / 'annotations' / 'train')
    copy_files(val_ann_paths, dataset_dir / 'annotations' / 'val')

    # Step 5: augment training set
    if configs['augment']:
        augment_dataset(dataset_dir / 'images' / 'train', img_ext, ann_ext)

    # Step 6: create Dataset Tensors
    train_dataset = tf.data.Dataset.list_files(str(dataset_dir / 'images' / 'train' / '*'))
    val_dataset = tf.data.Dataset.list_files(str(dataset_dir / 'images' / 'val' / '*'))

    # replace each string with a tuple that has elements (image_tensor, annotation_tensor, image_path_tensor)
    train_dataset = train_dataset.map(lambda x: parse_image(x, img_ext, ann_ext), num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.map(lambda x: parse_image(x, img_ext, ann_ext), num_parallel_calls=AUTOTUNE)

    BUFFER_SIZE = 48

    # shuffle the training dataset and batch it
    train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(configs['batch_size'])
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    # batch the validation dataset
    val_dataset = val_dataset.repeat()
    val_dataset = val_dataset.batch(1)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

    # determine number of steps to take in an epoch
    train_steps = len(list((dataset_dir / 'images' / 'train').iterdir())) // configs['batch_size']
    val_steps = len(list((dataset_dir / 'images' / 'val').iterdir())) // 1

    return {'train_dataset' : train_dataset, 'val_dataset' : val_dataset, 'train_steps' : train_steps, 'val_steps' : val_steps}


def create_train_val_inference_dataset(configs: dict) -> dict:

    data_dir: Path = configs['root_dir'] / 'data' / configs['dataset_name']
    img_ext = next(iter({file.suffix for file in (data_dir / 'images').iterdir()}))

    # Step 1: convert filenames to full paths
    train_paths = [data_dir / 'images' / (file + img_ext) for file in configs['training_set']]
    val_paths = [data_dir / 'images' / (file + img_ext) for file in configs['validation_set']]

    # Step 2: convert full paths list to tensor
    train_dataset = tf.data.Dataset.from_tensor_slices([str(path) for path in train_paths])
    val_dataset = tf.data.Dataset.from_tensor_slices([str(path) for path in val_paths])

    # Step 3: load images and batch them
    train_dataset = train_dataset.map(lambda x: parse_image(x, img_ext, None), num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.map(lambda x: parse_image(x, img_ext, None), num_parallel_calls=AUTOTUNE)

    train_dataset = train_dataset.batch(1)
    val_dataset = val_dataset.batch(1)

    return {'train_dataset' : train_dataset, 'train_paths' : train_paths, 'val_dataset' : val_dataset, 'val_paths' : val_paths}


def create_inference_dataset(configs: dict) -> dict[tf.Tensor, list[Path]]:
     # define path to images
    data_dir= configs['root_dir'] / 'data' / configs['dataset_name']
    data_paths = [path for path in data_dir.iterdir()]
    img_ext = next(iter(set(data_paths)))

    # create tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices([str(path) for path in data_paths])
    dataset = dataset.map(lambda x: parse_image(x, img_ext, None), num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(1)

    return {'dataset' : dataset, 'data_paths' : data_paths}


"""Miscellaneous functions"""


def split_data(configs: dict) -> Tuple[List[str], List[str]]:    
    # get current state
    data_dir = configs['root_dir'] / 'data' / configs['dataset_name'] / 'images'
    all_fns = [file.stem for file in data_dir.iterdir()]
    train_fns, val_fns = configs['training_set'], configs['validation_set']

    # Case 1: only training files provided
    if (train_fns is not None) and (val_fns is None):
        # split it with the given percentage
        if configs['auto_split']:
            random.shuffle(train_fns)
            train_upper = int(len(train_fns)*(1-configs['auto_split']))
            return train_fns[:train_upper], train_fns[train_upper:]
        # or use remaining files
        else:
            val_fns = os_sorted(list(set(all_fns) - set(train_fns)))
            return train_fns, val_fns
    
    # Case 2: only validation files provided
    elif (train_fns is None) and (val_fns is not None):
        # use remaining files
        train_fns = os_sorted(list(set(all_fns) - set(val_fns)))
        return train_fns, val_fns
    
    # Case 3: both training and validation files provided
    elif (train_fns is not None) and (val_fns is not None):
        # do nothing
        return train_fns, val_fns

    # Case 4: neither training nor validation files provided
    else:
        # split it with the given percentage (40% is default)
        random.shuffle(all_fns)
        train_upper = int(len(all_fns)*(1-configs['auto_split']))
        return all_fns[:train_upper], all_fns[train_upper:]


def copy_files(file_paths: list[Path], dest_dir: Path):
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)
       
    for path in file_paths:
        shutil.copy(path, dest_dir / path.name)


def augment_dataset(train_image_dir: Path, img_ext: str, ann_ext: str):
    # loop through training images
    for img in glob(str(train_image_dir / '*')):
        # replace images in path with annotations and image extension to annotation extension
        ann = re.sub('images', 'annotations', img)
        ann = re.sub(img_ext, ann_ext, ann)

        # augment image and annotation
        augment_single_image(Path(img))
        augment_single_image(Path(ann))


def augment_single_image(path : Path):
    # load file
    image = cv.imread(str(path))

    # perform transformations on image
    image_1 = path.parent / (path.stem + '_1' + path.suffix)    # original (just change its name)
    image_2 = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)          # rot90
    image_3 = cv.rotate(image, cv.ROTATE_180)                   # rot180
    image_4 = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)   # rot270
    image_5 = cv.flip(image, 1)                                 # xflip
    image_6 = cv.flip(image_2, 1)                               # rot90 + xflip
    image_7 = cv.flip(image_2, 0)                               # rot90 + yflip
    image_8 = cv.flip(image_3, 1)                               # rot180 + xflip

    # save augmentations
    path.rename(image_1)
    cv.imwrite(str(path.parent / (path.stem + '_2' + path.suffix)), image_2)
    cv.imwrite(str(path.parent / (path.stem + '_3' + path.suffix)), image_3)
    cv.imwrite(str(path.parent / (path.stem + '_4' + path.suffix)), image_4)
    cv.imwrite(str(path.parent / (path.stem + '_5' + path.suffix)), image_5)
    cv.imwrite(str(path.parent / (path.stem + '_6' + path.suffix)), image_6)
    cv.imwrite(str(path.parent / (path.stem + '_7' + path.suffix)), image_7)
    cv.imwrite(str(path.parent / (path.stem + '_8' + path.suffix)), image_8)


def parse_image(img_path: tf.Tensor, img_ext: str, ann_ext: str) -> tuple[tf.Tensor, tf.Tensor]:
    # read image and load it into 3 channels (pre-trained backbones require 3) and normalize it
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    
    try:
        # adjust path to lead to the corresponding annotation and load it
        ann_path = tf.strings.regex_replace(img_path, 'images', 'annotations')
        ann_path = tf.strings.regex_replace(ann_path, img_ext, ann_ext)
        annotation = tf.io.read_file(ann_path)
        annotation = tf.image.decode_png(annotation, channels=1)
        annotation = tf.cast(annotation, tf.float32) / 255.0
        return image, annotation
    except:
        # if ann_ext is None, we just want the image
        return image