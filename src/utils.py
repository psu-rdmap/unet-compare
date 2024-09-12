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


def inference(configs : dict, model : tf.keras.Model):
    """
    Inferences and saves every training/validation image after training
    
    Parameters
    ----------
    configs : dict
        Input configs given by the user
    model : tf.keras.Model
        Trained neural network
    """

    # base image source
    img_path = os.path.join(configs['root'], 'data/', configs['dataset_prefix'], 'images/')
    
    # train and val save destinations
    train_save_path = os.path.join(configs['root'], configs['results'], 'train_preds/')
    val_save_path = os.path.join(configs['root'], configs['results'], 'val_preds/')
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)

    # inference train images
    for fn in configs['train']:
        img_full_path = os.path.join(img_path, fn + configs['image_ext'])
        inference_single_image(img_full_path, model, train_save_path)

    # inference val images
    for fn in configs['val']:
        img_full_path = os.path.join(img_path, fn + configs['image_ext'])
        inference_single_image(img_full_path, model, val_save_path)


def inference_single_image(img_full_path : str, model : tf.keras.Model, img_save_path : str):
    """
    Loads, inferences, and saves an image with the given path
    
    Parameters
    ----------
    img_full_path : str
        Absolute path to an image used in the training process
    model : tf.keras.Model
        Trained neural network
    img_save_path : str
        Absolute path to the save destination
    """

    
    # get file path pieces
    _, fn_ext = os.path.split(img_full_path)
    fn, _ = os.path.splitext(fn_ext)
    
    # load image
    img = tf.io.read_file(img_full_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    
    # get prediction
    pred = model.predict(img, verbose = 2)

    # save pred
    pred_save_path = os.path.join(img_save_path, fn + '.png')
    save_img(pred_save_path, tf.squeeze(pred, axis=0))


def plot_results(configs : dict):
    """
    Loads training metrics from a single training loop, plots them, and saves it into the results directory

    Parameters
    ----------
    configs : dict
        Input configs given by the user
    """

    # paths
    results_path = os.path.join(configs['root'], configs['results'])
    metrics_path = os.path.join(results_path, 'metrics.csv')
    plot_save_path = os.path.join(results_path, 'metrics.png')

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
    results_path = os.path.join(configs['root'], configs['results'])
    loss_save_path = os.path.join(results_path, 'loss.png')
    metrics_save_path = os.path.join(results_path, 'metrics.png')

    # get fold directory names and sort them using natural sorting
    fold_dirs = os.listdir(results_path)
    fold_dirs = os_sorted(fold_dirs)

    # dict to hold dataframes for each fold
    all_metrics = []
    for fold in range(len(fold_dirs)):
        # get metrics from csv
        fold_metrics = pd.read_csv(os.path.join(results_path, fold_dirs[fold], 'metrics.csv'))

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
        ax.set_xlim([0, configs['num_epochs']])
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