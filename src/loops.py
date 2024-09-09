import models
import dataloader
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from os import mkdir
from os.path import join
import time
from utils import inference, plot_results, create_folds, cv_plot_results
import shutil
from keras import backend as K
import gc


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

    print('\nStarting training loop...\n')
    # start training 
    model.fit(
        dataset['train'],
        epochs=configs['num_epochs'],
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        validation_data=dataset['val'],
        callbacks=callbacks,
        verbose=1 # 1 = live progress bar, 2 = one line per epoch
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




    
    
    
