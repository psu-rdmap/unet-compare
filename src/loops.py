import models
import dataloader
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from os import mkdir, listdir
from os.path import join, split
import time
from utils import inference, plot_results, create_folds, cv_plot_results
import shutil
from keras import backend as K
import gc
import keras
from glob import glob
import numpy as np

def single(configs):
    # dataset
    print('\nCreating dataset...')
    time.sleep(2)
    dataset, train_steps, val_steps = dataloader.create_dataset(configs)
    print('\nTraining images: ', configs['train'])
    print('Validation images:', configs['val'])

    # model
    print('\nFetching and compiling model...')
    time.sleep(0.5)
    model = models.Decoder(configs).decoder
    
    # optimizer and loss
    model.compile(
        optimizer = Adam(learning_rate=configs['learning_rate']),
        loss = 'binary_crossentropy',
        metrics = ['accuracy', 'Precision', 'Recall']
    )
    
    # training callbacks
    callbacks = [
        CSVLogger(
            join(configs['root'], configs['results'], 'metrics.csv'), 
            separator=',', 
            append=False
        ),
        ModelCheckpoint(
            join(configs['root'], configs['results'], 'best.weights.h5'), 
            verbose=1, 
            save_best_only=True, 
            save_weights_only=True
        )
    ]
    if configs['early_stopping'] == True:
        callbacks.append(EarlyStopping(patience = configs['patience']))

    print('\nStarting training loop...\n')
    # start training 
    model.fit(
        dataset['train'],
        epochs=configs['num_epochs'],
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        validation_data=dataset['val'],
        callbacks=callbacks,
        verbose=2 # 1 = live progress bar, 2 = one line per epoch
    )

    print('\nInferencing image sets...\n')
    # inferences train/val sets
    inference(configs, model)

    print('\nPlotting metrics...')
    # plot results
    plot_results(configs)

    print('\nCleaning up...')
    # cleanup
    shutil.rmtree(join(configs['root'], 'dataset'))
    K.clear_session()
    gc.collect()

    if configs['training_loop'] == 'Single':
        print('\nDone.')


def cross_val(configs):    
    # generate split datasets for each fold
    train_sets, val_sets = create_folds(configs['train'], configs['num_folds'])
    
    # save original results directory
    top_level_results = configs['results']
    
    # loop through folds
    for fold in range(configs['num_folds']):
        # add train/val sets
        configs.update({'train' : train_sets[fold]})
        configs.update({'val' : val_sets[fold]})

        # create results directory for this fold
        results_dir = 'fold_' + str(fold+1)
        results_dir = join(configs['results'], results_dir)
        configs.update({'results' : results_dir})
        mkdir(join(configs['root'], results_dir))
        
        print()
        print('-'*20 + ' Fold {} '.format(fold+1) + '-'*20)

        # start single loop training
        single(configs)

        # redefine fold directory
        configs.update({'results' : top_level_results})

    print('\nPlotting CV Metrics...')
    time.sleep(1)
    cv_plot_results(configs)

    print('\nDone.')


def inference(configs):
    """
    Needs:
        Images
        Model
        Dest
    """ 

    # dataset
    print('\nLoading images...')
    time.sleep(2)

    # define path to images
    data_path = join(configs['root'], 'data/', configs['dataset_prefix'])
    fns = glob(data_path + '/*')

    # replace fns with loaded numpy arrays
    images = fns.copy()
    images = map(lambda x: dataloader.parse_inference_image(x, configs), images)
    images = list(images)

    # model
    print('\nFetching model and loading weights...')
    time.sleep(1)
    model = models.Decoder(configs).decoder
    model.load_weights(join(configs['root'], configs['weights_path']))

    print('\nGenerating predictions and saving them...\n')
    save_path = join(configs['root'], configs['results'])

    for i in range(len(fns)):
        pred = model.predict(images[i])
        pred = tf.squeeze(pred, axis=0)
        fn_ext = split(fns[i])[1]
        keras.utils.save_img(join(save_path, fn_ext), pred)

    print('\nDone.')
