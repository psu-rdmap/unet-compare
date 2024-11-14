"""
This module handles all accessory operations such as plotting and inference
"""

import os
import tensorflow as tf
from keras.preprocessing.image import save_img
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from natsort import os_sorted
from glob import glob


def parse_inference_image(img_path : str) -> tf.Tensor:
    """
    Given the full path to an image (/path/to/image.ext), load it into a tensor

    Parameters
    ----------
    img_path : str
        Full path to image file
    
    Returns
    -------
    Loaded image : tf.Tensor
    """

    # read image and load it into 3 channels (pre-trained backbones require 3)
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)

    # convert tensor objects to floats and normalize images ([0,255] -> [0.0, 1.0]) and return tensor
    return tf.cast(image, tf.float32) / 255.0


def save_preds(preds : tf.Tensor, save_fns : list):
    """
    Save each prediction from a tensor of predictions

    Parameters
    ----------
    preds : tf.Tensor
        Tensor of predictions with shape (N, H, W, 1) where N is the number of predictions
    save_fns : list
        List of corresponding image filenames
    """
    
    for idx, pred in enumerate(preds):
        save_img(save_fns[idx], pred)


def inference_ds(configs : dict, model : tf.keras.Model):
    """
    Inferences and saves every training/validation image after training
    
    Parameters
    ----------
    configs : dict
        Input configs given by the user
    model : tf.keras.Model
        Trained neural network
    """

    data_path = os.path.join(configs['root'], 'data/', configs['dataset_prefix'], 'images/')

    # define training and validation dataset
    train_ds = tf.data.Dataset.from_tensor_slices([data_path + img + configs['image_ext'] for img in configs['train']])
    val_ds = tf.data.Dataset.from_tensor_slices([data_path + img + configs['image_ext'] for img in configs['val']])

    # replace every image path in the training and validation directories with a loaded image and annotation pair
    train_ds = train_ds.map(parse_inference_image)
    val_ds = val_ds.map(parse_inference_image)

    # batch data
    train_ds = train_ds.batch(1)
    val_ds = val_ds.batch(1)

    # get predictions
    train_preds = model.predict(train_ds, verbose=1)
    val_preds = model.predict(val_ds, verbose=1)

    # define output directories
    train_save_dir = os.path.join(configs['results'], 'train_preds')
    os.mkdir(train_save_dir)
    train_save_fns = [os.path.join(train_save_dir, fn + '.png') for fn in configs['train']]

    val_save_dir = os.path.join(configs['results'], 'val_preds')
    os.mkdir(val_save_dir)
    val_save_fns = [os.path.join(val_save_dir, fn + '.png') for fn in configs['val']]

    # save train and val preds
    save_preds(train_preds, train_save_fns)
    save_preds(val_preds, val_save_fns)


def plot_results(configs : dict):
    """
    Loads training metrics from a single training loop, plots them, and saves it into the results directory

    Parameters
    ----------
    configs : dict
        Input configs given by the user
    """

    # paths
    metrics_path = os.path.join(configs['results'], 'metrics.csv')
    plot_save_path = os.path.join(configs['results'], 'metrics.png')

    # read metrics into dataframe
    metrics = pd.read_csv(metrics_path)

    # get num epochs, and redefine it offset by 1
    num_epochs = metrics['epoch'].count()
    metrics['epoch'] = metrics['epoch'] + 1

    # determine lowest val loss index (where val loss is closest to min(val_loss))
    best_idx = np.where(np.isclose(metrics['val_loss'], min(metrics['val_loss'])))[0]

    # add f1 columns to the dataframe
    metrics['f1'] = add_f1(metrics)
    metrics['val_f1'] = add_f1(metrics, val = True)

    # generate subplots
    fig, axs = plt.subplots(4, 1, figsize=(12,20))

    titles = ['BCE Loss', 'Precision', 'Recall', 'F1-Score']
    y_axes = ['loss', 'Precision', 'Recall', 'f1']

    for i in range(len(axs)):
        y1 = y_axes[i]
        y2 = 'val_' + y1

        # plot metric curve
        axs[i].plot(metrics['epoch'], metrics[y1], '-o',  label='Train')
        axs[i].plot(metrics['epoch'], metrics[y2], '-o', label='Val')
        
        # add point corresponding to lowest val loss on each curve
        axs[i].plot(best_idx + 1, metrics[y1].iloc[best_idx], 'D', color='purple')
        axs[i].plot(best_idx + 1, metrics[y2].iloc[best_idx], 'D', color='purple', label='Min Val Loss')
        
        # misc settings
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(titles[i])
        axs[i].set_xlim([1,num_epochs])
        if i == 0:
            axs[i].set_yscale('log')
        else:
            axs[i].set_yticks(ticks=np.arange(0,1.1,0.1))
        axs[i].grid(visible=True)
        axs[i].legend()     

    fig.savefig(plot_save_path, bbox_inches="tight")


def add_f1(metrics : pd.DataFrame, val = False) -> pd.DataFrame:
    """
    Calculates the f1-score element-wise given columns of the dataframe
    
    Parameters
    ----------
    metrics : pd.DataFrame
        Metrics from the csv file made while training
    val : bool
        Whether the metrics correspond to the validation (true) or training (false) set 
    
    Returns
    -------
    New metrics DataFrame column with f1 values : pd.DataFrame
    """

    if val == True:
        p = 'val_Precision'
        r = 'val_Recall'
    else:
        p = 'Precision'
        r = 'Recall'
    
    # if the denominator is 0, then f1=0, otherwise it is the harmonic mean
    return np.where(metrics[p] + metrics[r] == 0, 0, 2  * (metrics[p] * metrics[r]) / (metrics[p] + metrics[r]))


def cv_plot_results(configs : dict):
    """
    Loads in metrics for every fold, plots loss curves together, and plots statistics for each epoch

    Parameters
    ----------
    configs : dict
        Input configs given by the user
    """

    # paths
    loss_save_path = os.path.join(configs['results'], 'loss.png')
    metrics_save_path = os.path.join(configs['results'], 'metrics.png')

    # get fold directory names and sort them using natural sorting
    fold_dirs = glob(os.path.join(configs['results'], 'fold_*/'))
    fold_dirs = os_sorted(fold_dirs)

    # dict to hold dataframes for each fold
    all_metrics = []
    for fold in range(len(fold_dirs)):
        # get metrics from csv
        fold_metrics = pd.read_csv(os.path.join(fold_dirs[fold], 'metrics.csv'))

        # add f1 columns to the dataframe
        fold_metrics['f1'] = add_f1(fold_metrics)
        fold_metrics['val_f1'] = add_f1(fold_metrics, val = True)

        # convert to np array and add metrics array to list 
        fold_metrics_np = fold_metrics.to_numpy()
        all_metrics.append(fold_metrics_np)

    # stack all metrics arrays along a new 3d axis
    all_metrics = np.stack(all_metrics, axis=0)

    # get epochs list (always the same)
    epochs = all_metrics[0, :, 0].astype(int) + 1

    # plot loss curves together on two separate plots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    for fold in range(configs['num_folds']):
        # train/val losses
        train_loss = all_metrics[fold, :, 4]
        val_loss = all_metrics[fold, :, 8]

        # plot losses
        axs[0].plot(epochs, train_loss, label = 'Fold {}'.format(fold+1))
        axs[1].plot(epochs, val_loss, label = 'Fold {}'.format(fold+1))

    # formatting
    axs[0].set_ylabel('Train Loss (BCE)')
    axs[1].set_ylabel('Val Loss (BCE)')
    for ax in axs:
        ax.set_yscale('log')   
        ax.set_xlim([1, configs['num_epochs']])
        ax.set_xlabel('Epoch')
        ax.grid(visible=True)
        ax.legend()

    fig.savefig(loss_save_path, bbox_inches="tight")

    # get mean and stdev across all folds
    num_metrics = np.shape(all_metrics)[-1]
    metrics_mean = np.zeros((configs['num_epochs'], num_metrics))
    metrics_std = np.zeros((configs['num_epochs'], num_metrics))

    for metric in range(num_metrics):
        for epoch in range(configs['num_epochs']):
            metrics_mean[epoch, metric] = np.mean(all_metrics[:, epoch, metric])
            metrics_std[epoch, metric] = np.std(all_metrics[:, epoch, metric])

    # plot averaged metrics with std as error bars
    fig, axs = plt.subplots(4, 1, figsize=(12, 20))

    # settings specific to each plot 
    titles = ['BCE Loss', 'Precision', 'Recall', 'F1-Score']
    train_metrics_idcs = [4, 1, 2, 9]
    val_metrics_idcs = [8, 5, 6, 10]

    for i in range(len(axs)):
        # means and stds
        train_mean = metrics_mean[:, train_metrics_idcs[i]]
        val_mean = metrics_mean[:, val_metrics_idcs[i]]

        train_std = metrics_std[:, train_metrics_idcs[i]]
        val_std = metrics_std[:, val_metrics_idcs[i]]

        # plot mean
        axs[i].errorbar(epochs, train_mean, yerr=train_std, fmt='-o', capsize=3, capthick=1, label='Train')
        axs[i].errorbar(epochs, val_mean, yerr=val_std, fmt='-o', capsize=3, capthick=1, label='Val')

        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(titles[i])
        axs[i].set_xlim([1, configs['num_epochs']])
        if i == 0:
            axs[i].set_yscale('log')
        else:
            axs[i].set_yticks(ticks=np.arange(0,1.1,0.1))
        axs[i].grid(visible=True)
        axs[i].legend()

    fig.savefig(metrics_save_path, bbox_inches="tight")


def create_folds(img_list : list, num_folds : int) -> tuple[list, list]:
    """
    Given images and the number of folds, create training/validation sets with the most even distribution possible

    Parameters
    ----------
    img_list : list
        List of filenames that will be used for cross-validation
    num_folds : int
        Number of training/validation sets to generate
    
    Returns
    -------
    Training set for each fold : list
    Validation set for each fold : list 
    """

    # number of validation images to be held out in each fold
    num_val = np.zeros(num_folds)

    # randomly shuffle the image list given a numpy seed to prevent sequence bias
    np.random.seed(203)
    img_list = np.random.permutation(img_list)

    # determine number of hold out images in each fold
    for i in range(num_folds):
        # start with integer quotient
        num_val[i] = math.floor(len(img_list) / num_folds)
        # distribute the remainder evenly among the first folds   
        if i < (len(img_list) % num_folds):
            num_val[i] += 1
    
    # convert number of hold out images to indicies
    running_sum = np.cumsum(num_val)
    running_sum = np.insert(running_sum, 0, 0)
    lower_idxs = running_sum[:-1].astype(int)
    upper_idxs = running_sum[1:].astype(int)

    # save train/val sets as elements of a list
    train_sets, val_sets = [0]*num_folds, [0]*num_folds
    for i in range(num_folds):
        # bounds
        low = lower_idxs[i]
        up = upper_idxs[i]

        # fold val set
        val_sets[i] = img_list[low:up]

        # fold train set is the complement
        train_sets[i] = np.delete(img_list, np.arange(low, up)).tolist()

    return train_sets, val_sets