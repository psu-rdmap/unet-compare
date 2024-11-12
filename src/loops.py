"""
This module handles the training/inference process
"""

import models
import dataloader
import utils
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from os import mkdir, listdir
from os.path import join, split
import time
import shutil
from keras import backend as K
import gc
import keras
from glob import glob
import numpy as np


def single_loop(configs : dict):
    """
    Runs a single epoch-based training loop

    Parameters
    ----------
    configs : dict
        Input configs provided by the user
    """
    
    # get dataset
    print('\nCreating dataset...')
    time.sleep(2)
    dataset, train_steps, val_steps = dataloader.create_dataset(configs)
    print('\nTraining images: ', configs['train'])
    print('Validation images:', configs['val'])

    # get model
    print('\nLoading model...')
    time.sleep(0.5)
    if configs['model_path'] is not None:
        model = keras.load_model(configs['model_path'])
    else:
        model = models.UNet(configs)
        model.compile(optimizer = Adam(learning_rate=configs['learning_rate']), loss = 'binary_crossentropy', metrics = ['accuracy', 'Precision', 'Recall'])
    
    callbacks = [
        CSVLogger(join(configs['results'], 'metrics.csv'), separator=',', append=False),
        ModelCheckpoint(join(configs['results'], 'best.model.keras'), verbose=1, save_best_only=configs['save_best_only'], save_weights_only=False)
    ]
    if configs['early_stopping']:
        callbacks.append(EarlyStopping(patience = configs['patience']))
    
    # training
    print('\nStarting training loop...\n')
    model.fit(
        dataset['train'],
        epochs=configs['num_epochs'],
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        validation_data=dataset['val'],
        callbacks=callbacks,
        verbose=2 # 1 = live progress bar, 2 = one line per epoch
    )

    # load model corresponding to minimum val loss
    if configs['save_best_only']:
        model = keras.saving.load_model(join(configs['results'], 'best.model.keras'))

    # inferences train/val sets and save results into results
    print('\nInferencing image sets...')
    utils.inference_ds(configs, model)

    # plot loss, precision, recall, and f1
    print('\nPlotting metrics...')
    utils.plot_results(configs)

    # remove training dataset and clear memory
    print('\nCleaning up...')
    shutil.rmtree(join(configs['root'], 'dataset'))
    K.clear_session()
    gc.collect()

    if configs['training_loop'] == 'Single':
        print('\nDone.')
        

def cross_val_loop(configs : dict):    
    """
    Runs single training loop many times on different hold out sets

    Parameters
    ----------
    configs : dict
        Input configs provided by the user
    """
    
    # determine all training and val set combinations given the number of folds
    train_sets, val_sets = utils.create_folds(configs['train'], configs['num_folds'])
    
    # save original results directory
    top_level_results = configs['results']
    
    # loop through folds
    for fold in range(configs['num_folds']):
        # update train/val sets
        configs.update({'train' : train_sets[fold]})
        configs.update({'val' : val_sets[fold]})

        # create results directory for this fold
        results_dir = 'fold_' + str(fold+1)
        results_dir = join(configs['results'], results_dir)
        configs.update({'results' : results_dir})
        mkdir(results_dir)
        
        print()
        print('-'*30 + ' Fold {} '.format(fold+1) + '-'*30)

        # train fold
        single_loop(configs)

        # reset results directory for next fold
        configs.update({'results' : top_level_results})

    # plot metrics over all folds
    print('\nPlotting CV Metrics...')
    time.sleep(1)
    utils.cv_plot_results(configs)

    print('\nDone.')


def inference_loop(configs : dict):
    """
    Feeds images into a trained model and saves predictions

    Parameters
    ----------
    configs : dict
        Input configs provided by the user
    """

    print('Loading images...')
    time.sleep(2)

    # define path to images and retrieve filenames
    data_path = join(configs['root'], 'data/', configs['dataset_prefix'])
    ds_fns = listdir(data_path)

    # create tensorflow ds
    ds = [join(data_path, fn) for fn in ds_fns]
    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = ds.map(utils.parse_inference_image)

    # build model and load weights
    print('\nLoading model...')
    time.sleep(1)
    model = keras.saving.load_model(configs['model_path'])

    # create and save predictions
    print('\nGenerating predictions and saving them...\n')
    preds = model.predict(ds, verbose=2)
    save_fns = [join(configs['results'], split(fn)[0], '.png') for fn in ds_fns]
    utils.save_preds(preds, save_fns)

    print('\nDone.')
