from glob import glob
from os import mkdir, listdir
from os.path import join
import shutil
import tensorflow as tf

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

    # split dataset
    data = configs['data']

    # copy images into dataset directory
    populate_dstree(configs['root'], data)

    # augment dataset
    augment_dataset()

    # create a Tensorflow Dataset object
    return define_dataset(train_val_paths, data)


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


def augment_dataset():
    """
    NOT IMPLIMENTED YET


    # using the root path to a directory containing images (e.g. dataset/images/train/)
    #     also using the list of images in said directory
    #     augment and save them as the 'save_format' type (e.g. jpg or png)
    for i in image_list:
        # extract file name without extension
        base = splitext(i)[0]

        # open image
        im = join(root, i)
        img = Image.open(im)
        
        # perform transformations on image
        img_1 = img
        img_2 = img.rotate(90)
        img_3 = img.rotate(180)
        img_4 = img.rotate(270)
        img_5 = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_6 = img_2.transpose(Image.FLIP_TOP_BOTTOM)
        img_7 = img.transpose(Image.FLIP_TOP_BOTTOM)
        img_8 = img_2.transpose(Image.FLIP_TOP_BOTTOM)

        # save augmentations
        img_1.save(join(root, base + "_1." + save_format))
        img_2.save(join(root, base + "_2." + save_format))
        img_3.save(join(root, base + "_3." + save_format))
        img_4.save(join(root, base + "_4." + save_format))
        img_5.save(join(root, base + "_5." + save_format))
        img_6.save(join(root, base + "_6." + save_format))
        img_7.save(join(root, base + "_7." + save_format))
        img_8.save(join(root, base + "_8." + save_format))

        # delete image object
        img.close()

        # remove original image and label
        remove(im)
    """
    pass


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
    STEPS_PER_EPOCH : int
        Number of steps to take when training for a single pass over all available training data
    VALIDATION_STEPS : int
        Number of steps to take during evaluation of the validation set after an epoch has completed
   
    """
    
    # get size of training and validation sets
    TRAINSET_SIZE = len(listdir(train_val_paths['train_path']))
    VALSET_SIZE = len(listdir(train_val_paths['val_path']))

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
    STEPS_PER_EPOCH = TRAINSET_SIZE // data_cfgs['batch_size']
    VALIDATION_STEPS = VALSET_SIZE // 1

    return dataset, STEPS_PER_EPOCH, VALIDATION_STEPS


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