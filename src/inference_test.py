import argparse
import os, sys
from os.path import join, isdir, splitext
from os import mkdir, listdir
import utils
import tensorflow as tf
import PIL
from keras.preprocessing.image import save_img
import numpy as np
import re

AUTOTUNE = tf.data.AUTOTUNE

# -----------Obtaining User Specifications-----------

parser = argparse.ArgumentParser(description='Generate CV Predictions')

parser.add_argument('--image_size', default=1024, type=int,
                    help='square image size')
parser.add_argument('--model', help='model architecture', default='unet', type=str)
parser.add_argument('--checkpoint_path', help='path to model checkpoint',
                    default='/storage/group/xvw5285/default/UNET/TRAINING/', type=str)
parser.add_argument('--results', help='path to contain inferences',
                    default='results', type=str)
parser.add_argument('--images', help='path to images to inference', 
                    default='test_images/', type=str)
parser.add_argument('--root', default='/storage/group/xvw5285/default/UNET/TRAINING/')

args = parser.parse_args()

# ---------------------------------------------------

# supress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYCARET_CUSTOM_LOGGING_LEVEL'] = 'CRITICAL'


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
sys.path.append(args.root)

def main():
    # get image filenames
    image_filenames = listdir(args.images)
    print('Images:', image_filenames)
    
    # create results directory
    if not isdir(args.results):
        mkdir(args.results)
    
    # get the model and load weights
    model = utils.get_model(args.model, args.image_size, l2_reg=0)
    model.load_weights(join(args.root, args.checkpoint_path))
    
    # loop through images and save predictions
    for img_fn in image_filenames:
        # inference image and save into results dir
        inference(args.images, 'jpg', img_fn, model, args.results)

def inference(path_to, img_ext, fn, model, out_dir):
    # open each image as numpy array and inference it
    fn_path = join(path_to, fn)
    img = tf.io.read_file(fn_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img,axis=0)
    pred = model.predict(img, verbose = 2)
    out_path = join(out_dir, re.sub(img_ext, 'png', fn))

    save_img(out_path, np.squeeze(pred, axis=0))

if __name__ == "__main__":
    main()
