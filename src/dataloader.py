from glob import glob
from os import mkdir, listdir, remove
from os.path import join, split, splitext
import shutil
import tensorflow as tf
import random
import re
import cv2

random.seed(229)
AUTOTUNE = tf.data.AUTOTUNE
tf.random.set_seed(3051)

def create_dataset(configs):
    """
    Create the training dataset and return a Tensorflow Dataset instance of it

    Parameters:
    ---------- 
    configs : dict
        Dictionary containing all data necessary to find the data and build the dataset
    
    Returns:
    --------
    Training and validation Dataset instances : dict
    """
    
    # create directory structure
    ds_path = join(configs['root'], 'dataset')
    train_val_paths = create_dstree(ds_path)

    # if auto_split is true, this function will split the data into train/val sets
    configs = split_data(configs)

    # copy images into dataset directory
    populate_dstree(configs['root'], configs['data'])

    # augment dataset
    augment_dataset(configs, train_val_paths['train'])

    # create a Tensorflow Dataset object
    return define_dataset(train_val_paths, configs['data'])


def create_dstree(dest):
    """
    Create the directory tree given the top-dataset directory path

    Parameters:
    ---------- 
    dest : str
        Desired path to dataset
    
    Returns:
    --------
    Full paths to image train and val sets : dict
    """

    # define directories
    image_dir = join(dest, 'images')
    annotation_dir = join(dest, 'annotations')
    image_train_dir = join(image_dir, 'train')
    image_val_dir = join(image_dir, 'val')
    annotation_train_dir = join(annotation_dir, 'train')
    annotation_val_dir = join(annotation_dir, 'val')

    # run through all directories defined in the function and make them
    for _, dir in locals().items():
        mkdir(dir)

    return {"train_path" : image_train_dir, "val_path" : image_val_dir}


def populate_dstree(root, data_cfgs):
    """
    Copy files from data directory to empty dataset tree

    Parameters:
    ---------- 
    root : str
        Root path of the project
    data_cfgs : dict
        Dataset creation data (dataset prefix, filenames, extensions, etc.)
    """

    # file sets
    train_images = [image + data_cfgs['image_ext'] for image in data_cfgs['train']]
    val_images = [image + data_cfgs['image_ext'] for image in data_cfgs['val']]

    train_annotations = [annotation + data_cfgs['annotation_ext'] for annotation in data_cfgs['train']]
    val_annotations = [annotation + data_cfgs['annotation_ext'] for annotation in data_cfgs['val']]

    # source directories
    image_source = join(root, 'data/', data_cfgs['dataset_prefix'], 'images/')
    annotation_source = join(root, 'data/', data_cfgs['dataset_prefix'], 'annotations/')

    # dest directories
    image_train_dest = join(root, 'dataset/images/train')
    image_val_dest = join(root, 'dataset/images/val')

    annotation_train_dest = join(root, 'dataset/annotations/train')
    annotation_val_dest = join(root, 'dataset/annotations/val')

    # populate directories
    copy_files(train_images, image_source, image_train_dest)
    copy_files(val_images, image_source, image_val_dest)

    copy_files(train_annotations, annotation_source, annotation_train_dest)
    copy_files(val_annotations, annotation_source, annotation_val_dest)


def augment_dataset(configs, image_train_path):
    """
    Replace each image and annotation with eight geometric augmentations

    Parameters:
    ---------- 
    configs : dict
        Dictionary with relevant data information (file extensions, etc.)
    image_train_path : str
        Full path to training images in created dataset directory 
    """

    # loop through training images (use glob to get full path to each image) 
    for img in glob(join(image_train_path, '*')):
        # replace images in path with annotations and image extension to annotation extension
        ann = re.sub('images', 'annotations', img)
        ann = re.sub(configs['data']['image_ext'], configs['data']['annotation_ext'], ann)

        # augment image and annotation
        augment_single_image(img)
        augment_single_image(ann)

    def augment_single_image(file_full_path):
        # get file path pieces
        path, fn_ext = split(file_full_path)
        fn, ext = splitext(fn_ext)

        # load file
        file = cv2.imread(file_full_path)
        
        # perform transformations on file
        file_1 = file                                              # original
        file_2 = cv2.rotate(file, cv2.ROTATE_90_CLOCKWISE)         # rot90
        file_3 = cv2.rotate(file, cv2.ROTATE_180)                  # rot180
        file_4 = cv2.rotate(file, cv2.ROTATE_90_COUNTERCLOCKWISE)  # rot270
        file_5 = cv2.flip(file, 1)                                 # xflip
        file_6 = cv2.flip(file_2, 1)                               # rot90 + xflip
        file_7 = cv2.flip(file_2, 0)                               # rot90 + yflip
        file_8 = cv2.flip(file_3, 1)                               # rot90 + xflip

        # save augmentations
        cv2.imwrite(join(path, fn + '_1', ext), file_1)
        cv2.imwrite(join(path, fn + '_2', ext), file_2)
        cv2.imwrite(join(path, fn + '_3', ext), file_3)
        cv2.imwrite(join(path, fn + '_4', ext), file_4)
        cv2.imwrite(join(path, fn + '_5', ext), file_5)
        cv2.imwrite(join(path, fn + '_6', ext), file_6)
        cv2.imwrite(join(path, fn + '_7', ext), file_7)
        cv2.imwrite(join(path, fn + '_8', ext), file_8)

        # remove original image and label
        remove(file_full_path)


def define_dataset(train_val_paths, data_cfgs):
    """
    Use created dataset to define Tensorflow Dataset instances for the training and validation sets

    Parameters:
    ---------- 
    data_cfgs : dict
        Dictionary containing all data necessary to find the data and build the dataset
    
    Returns:
    --------
    dataset : dict
        Dictionary of training and validation instantiated Dataset objects
    train_steps : int
        Number of steps to take when training for a single pass over all available training data
    val_steps : int
        Number of steps to take during evaluation of the validation set after an epoch has completed
    """
    
    # get size of training and validation sets
    train_size = len(listdir(train_val_paths['train_path']))
    val_size = len(listdir(train_val_paths['val_path']))

    # initialize Dataset objects with a list of filenames
    train_dataset = tf.data.Dataset.list_files(train_val_paths['train_path'] + '/*')
    val_dataset = tf.data.Dataset.list_files(train_val_paths['val_path'] + '/*')
    
    # replace every image path in the training and validation directories with a loaded image and annotation pair
    train_dataset = train_dataset.map(lambda x: parse_image(x, data_cfgs), num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.map(lambda x: parse_image(x, data_cfgs), num_parallel_calls=AUTOTUNE)

    BUFFER_SIZE = 48

    # define dict to contain Dataset instances
    dataset = {"train": train_dataset, "val": val_dataset}

    # shuffle the training dataset and batch it
    dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE)
    dataset['train'] = dataset['train'].repeat()
    dataset['train'] = dataset['train'].batch(data_cfgs['batch_size'])
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

    # batch the validation dataset
    dataset['val'] = dataset['val'].repeat()
    dataset['val'] = dataset['val'].batch(1)
    dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

    # determine number of steps to take in an epoch
    train_steps = train_size // data_cfgs['batch_size']
    val_steps = val_size // 1

    print(type(train_steps), type(val_steps))

    return dataset, train_steps, val_steps


def copy_files(files, source, dest):
    """
    Copies all provided files from a source directory to a destination directory

    Parameters:
    ---------- 
    files : list
        Filenames in source directory to be copied
    source : str
        Path to current location of files
    dest : str
        Path to where files are to be copied to
    """

    for file in files:
        src = join(source, file)
        dst = join(dest, file)
        shutil.copy(src, dst)


@tf.function
def parse_image(img_path, data_cfgs):
    """
    Load an image and its annotation then resize it and normalize it

    Parameters:
    ---------- 
    img_path : tf.str
        Path to a given image to be loaded
    data_cfgs : dict
        Dictionary containing all info necessary to load and pre-process an image
    
    Returns:
    --------
    Loaded image/annotation pair : dict
    """
    
    # read image and load it into 3 channels (pre-trained backbones require 3)
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)

    # adjust path to lead to the corresponding annotation and load it
    annotation_path = tf.strings.regex_replace(img_path, "images", "annotations")
    annotation_path = tf.strings.regex_replace(annotation_path, data_cfgs['image_ext'], data_cfgs['annotation_ext'])
    annotation = tf.io.read_file(annotation_path)
    annotation = tf.image.decode_png(annotation, channels=1)

    # resize image and annotation
    resize_shape = tf.convert_to_tensor(data_cfgs['input_shape'], dtype=tf.int32)
    image = tf.image.resize(image, resize_shape, method=tf.image.ResizeMethod.LANCZOS3, preserve_aspect_ratio=True)
    annotation = tf.image.resize(annotation, resize_shape, method=tf.image.ResizeMethod.LANCZOS3, preserve_aspect_ratio=True)

    # convert tensor objects to floats and normalize images ([0,255] -> [0.0, 1.0])
    image = tf.cast(image, tf.float32) / 255.0
    annotation = tf.cast(annotation, tf.float32) / 255.0

    # return a dict containing the loaded image/annotation pair
    return {'image': image, 'annotation': annotation}


def split_data(configs):
    """
    Add training and validation sets to configs if necessary

    Parameters:
    ---------- 
    configs : dict
        Information required to generate training and validation sets
    
    Returns:
    --------
    configs : dict
        Updated configs w/ train/val sets
    """
    
    # case 1: auto_split is on and training filenames are provided
    if (configs['auto_split'] == True) and ('train' in configs['data']):       
        # randomly select training and validation subsets
        train_fns, val_fns = auto_split(configs['data']['train'], configs['data']['val_hold_out'])

        # overwrite training filename list and add new val filename list
        configs['data']['train'] = train_fns
        configs['data'].update({'val' : val_fns})

    # case 2: auto_split is on and no training filenames are provided
    elif configs['auto_split'] == True:
        # get fns from image file source
        fns_path = join(configs['root'], 'data/', configs['data']['dataset_prefix'], 'images/')
        fns = listdir(fns_path)

        # randomly select training and validation subsets
        train_fns, val_fns = auto_split(fns, configs['data']['val_hold_out'])

        # add train and val filename lists
        configs['data'].update({'train' : train_fns})
        configs['data'].update({'val' : val_fns})

    # case 3: data is already split and provided by the user
    else:
        pass

    return configs
    

def auto_split(fns, val_hold_out):
    """
    Randomly select subsets of the fns list for training and validation

    Parameters:
    ---------- 
    fns : list
        List of filenames to be split
    val_hold_out : dict
        Fraction of files to be held out for validation 

    Returns:
    --------
    train_fns : list
        Filenames to be used for training
    val_fns : list
        Filenames to be used for validation
    """
    
    # shuffle images to prevent sequence bias
    random.shuffle(fns)

    # upper/lower bounds
    train_lower = 0
    train_upper = int(len(fns)*val_hold_out)
    
    val_lower = train_upper
    val_upper = len(fns)

    # define and return new filenames sets
    train_fns = fns[train_lower : train_upper]
    val_fns = fns[val_lower : val_upper]

    return train_fns, val_fns