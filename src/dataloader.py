from glob import glob
from os.path import join
import tensorflow as tf

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

def dataloader(dataset_root, BATCH_SIZE, img_ext, lab_ext):
    """
    input: dataset root
    output: tensorflow dataset
    """

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
