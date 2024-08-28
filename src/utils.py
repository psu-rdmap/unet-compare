import models
from PIL import Image
from os import remove
from os.path import join, splitext
import shutil

def get_model(model_type, IMG_SIZE, l2_reg, batchnorm=True):
    if model_type == "unet":
        model = models.unet(IMG_SIZE, l2_reg, batchnorm)

    elif model_type == "unet_vgg":
        model = models.unet_vgg16(IMG_SIZE, l2_reg, batchnorm)

    elif model_type == "unet_eff":
        model = models.unet_eff(IMG_SIZE, l2_reg, batchnorm)

    elif model_type == "unetpp":
        model = models.unetpp(IMG_SIZE, l2_reg, batchnorm)
          
    elif model_type == "unetpp_vgg":
        model = models.unetpp(IMG_SIZE, l2_reg, batchnorm)

    elif model_type == "unetpp_eff":
        model = models.unetpp_eff(IMG_SIZE, l2_reg, batchnorm)

    else:
        raise ValueError("Incorrect model type specified")

    return model


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


def parse_input(tr_path, ds_path):
    # input:
        # file names with training details and dataset filenames
    # output:
        # dict with training parameters and data filenames
    
    # create keyword namespace
    info = dict(image_size = 1024,
                         batch_size = 1,
                         learning_rate = 1e-3,
                         l2_regularization = 1e-3,
                         max_epochs = 100,
                         early_stopping = True,
                         patience = 20,
                         cross_validation = False,
                         num_folds = None,
                         restart_fold = None,
                         model = 'unet',
                         dataset_identifier = 'bubble',
                         batchnorm = True,
                         learning_scheduler = 'step',
                         decay_fact = 0.5,
                         decay_freq=10,
                         root = '/storage/group/xvw5285/default/UNET/TRAINING/',
                         images_ext = 'jpg', 
                         labels_ext = 'png', 
                         images_path = 'images', 
                         labels_path = 'labels',
                         augment = True,
                         results_dir_path = '') 
    
    # get kwargs
    with open(tr_path) as reader:
        lines = reader.readlines()
        
        for line in lines:
            # assume line is a kwarg pair
            kw, arg = line.split()

            # check if kw is valid in dict
            if kw in info.keys():
                info[kw] = arg
            else:
                raise Exception("Invalid keyword: {}".format(kw))
    
    # leave out val filenames if cv was toggled
    if eval(info['cross_validation']):
        ds_fns = dict(train = [])
    else:
        ds_fns = dict(train = [], val = [])
    
    with open(ds_path) as reader:
        lines = reader.readlines()

        for line in lines:
            line = line.strip()
            # if train (or val)
            if eval(info['cross_validation']) and line == 'val':
                raise Exception("Do not specify a validation set if you are using cross validation")
            elif line in ds_fns.keys():
                fn_set = line
            # otherwise a filename
            else:
                ds_fns[fn_set].append(line)
            
    return info, ds_fns
