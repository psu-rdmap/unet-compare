"""
Aiden Ochoa, 4/2025, RDMAP PSU Research Group
This module handles all accessory operations such as plotting
"""

from keras.preprocessing.image import save_img
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math, json
from natsort import os_sorted
from glob import glob
from pathlib import Path


def save_preds(preds : list[np.array], save_paths : list[Path]):
    """Save inference predictions from model.predict()"""
    
    for i, pred in enumerate(preds):
        save_img(str(save_paths[i]), pred)


def print_save_configs(configs : dict):
    """Creates the results directory, prints the input configs after validation, and saves a copy to results"""
    
    # create top-level results directory
    configs['results_dir'].mkdir()
    
    # print input to user for confirmation
    print('-'*60 + ' User Input ' + '-'*60)
    for key, val in configs.items():
        print(key + ':', val)
    print('-'*132)

    # save configs into results dir for reference
    with open(configs['results_dir'] / 'configs.json', 'w') as con:
        # make Path objects strings for serialization
        configs['root_dir'] = str(configs['root_dir'])
        configs['results_dir'] = str(configs['results_dir'])
        json.dump(configs, con)


def plot_results(configs : dict):
    """Loads training metrics from a single training loop, plots them, and saves it to the results directory"""

    # paths
    metrics_path = configs['results_dir'] / 'metrics.csv'
    plot_save_path = configs['results_dir'] / 'metrics.png'

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

    fig.savefig(str(plot_save_path), bbox_inches="tight")


def add_f1(metrics : pd.DataFrame, val = False) -> pd.DataFrame:
    """Calculates the f1-score element-wise given columns of a Pandas metrics dataframe"""

    if val == True:
        p = 'val_Precision'
        r = 'val_Recall'
    else:
        p = 'Precision'
        r = 'Recall'
    
    # if the denominator is 0, then f1=0, otherwise it is the harmonic mean
    return np.where(metrics[p] + metrics[r] == 0, 0, 2  * (metrics[p] * metrics[r]) / (metrics[p] + metrics[r]))


def cv_plot_results(configs : dict):
    """Loads in metrics from every cross validation fold, plots loss curves together, and plots statistics for each epoch"""

    # paths
    loss_save_path = configs['results_dir'] / 'loss.png'
    metrics_save_path = configs['results_dir'] / 'metrics.png'

    # get fold directory names and sort them using natural sorting
    fold_dirs = glob(str(configs['results_dir'] / 'fold_*'))
    fold_dirs = os_sorted(fold_dirs)

    # dict to hold dataframes for each fold
    all_metrics = []
    for fold in range(len(fold_dirs)):
        # get metrics from csv
        fold_metrics = pd.read_csv(str(Path(fold_dirs[fold]) / 'metrics.csv'))

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

    fig.savefig(str(loss_save_path), bbox_inches="tight")

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

    fig.savefig(str(metrics_save_path), bbox_inches="tight")


def create_folds(img_list : list, num_folds : int) -> tuple[list, list]:
    """Given images and the number of folds, create training/validation sets with the most even distribution possible"""

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