from glob import glob
from os import mkdir
from os.path import join
#import tensorflow as tf
#AUTOTUNE = tf.data.AUTOTUNE

def create_dataset(configs):
    # create directory structure
    ds_path = join(configs['root'], 'dataset')
    create_dstree(ds_path)

    # copy images into dataset directory
    data_path = join(configs['root'], configs['dataset_prefix'])
    populate_dstree(data_path, ds_path)



def create_dstree(dest):
    """
    Create the directory tree given the top-dataset directory path

    Parameters:
    ---------- 
    dest : str
        Desired path to dataset
      
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

def populate_dstree(source, dest):
    """
    Copy files from data directory to empty dataset tree

    Parameters:
    ---------- 
    source : str
        Path to images and annotations
    dest : str
        Root of dataset tree
      
    """

    


configs = dict(
    root = '/home/aidenochoa/unet-compare/',
    dataset_prefix = 'gb',
)

create_dataset(configs)

"""
def dataloader(configs):
    



    # raw data paths
    dataset_path = join(dataset_root, "images/")
    training_data = "train/"
    val_data = "val/"

    TRAINSET_SIZE = len(glob(join(dataset_path, training_data) + f"*.{img_ext}"))
    VALSET_SIZE = len(glob(join(dataset_path, val_data) + f"*.{img_ext}"))

    train_dataset = tf.data.Dataset.list_files(join(dataset_path, training_data) + f"*.{img_ext}", seed=SEED)
    train_dataset = train_dataset.map(lambda x: parse_image(x, img_ext, lab_ext))

    val_dataset = tf.data.Dataset.list_files(join(dataset_path, val_data) + f"*.{img_ext}", seed=SEED)
    val_dataset = val_dataset.map(lambda x: parse_image(x, img_ext, lab_ext))

    BUFFER_SIZE = 48

    dataset = {"train": train_dataset, "val": val_dataset}

    # -- Train Dataset --#
    dataset['train'] = dataset['train'].map(load_image, num_parallel_calls=AUTOTUNE)
    dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
    dataset['train'] = dataset['train'].repeat()
    dataset['train'] = dataset['train'].batch(BATCH_SIZE)
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

    # -- Validation Dataset --#
    dataset['val'] = dataset['val'].map(load_image)
    dataset['val'] = dataset['val'].repeat()
    dataset['val'] = dataset['val'].batch(BATCH_SIZE)
    dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

    STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
    VALIDATION_STEPS = VALSET_SIZE // BATCH_SIZE

    return dataset, STEPS_PER_EPOCH, VALIDATION_STEPS

def parse_image(img_path, img_ext, lab_ext):
    # input: file path to image
    # output: image (greyscale) and corresponding mask as dictionary value

    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)

    mask_path = tf.strings.regex_replace(img_path, "images", "labels")
    mask_path = tf.strings.regex_replace(mask_path, img_ext, lab_ext)
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)

    return {'image': image, 'segmentation_mask': mask}

@tf.function
def load_image(datapoint: dict) -> tuple:
    # convert to tensors and normalize images
    input_image = tf.cast(datapoint['image'], tf.float32) / 255.0
    input_mask = tf.cast(datapoint['segmentation_mask'], tf.float32) / 255.0
    return input_image, input_mask

# function for augmenting a given dataset
def augment(root, image_list, save_format):
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


# function for copying images/labels into the generated dataset
def copy_files(filename_set, in_dir, out_dir):
    # using the filenames in the list 'filename_set'
    # copy the file from the 'in_dir' into the 'out_dir'
    for file in filename_set:
        src = join(in_dir, file)
        dst = join(out_dir, file)
        shutil.copy(src, dst)
"""