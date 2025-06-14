"""
Aiden Ochoa, 4/2025, RDMAP PSU Research Group
This module handles all training and inference operations. It has __main__
"""

import argparse, json, shutil, multiprocessing, gc, os, keras
import input_validator, dataloader, models, utils
from keras.api.saving import load_model
from keras.api.optimizers import Adam
from keras.api.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras import backend as K
from pathlib import Path
from natsort import os_sorted
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# get config file from input
parser = argparse.ArgumentParser(description='U-Net Training')
parser.add_argument('configs', type=str, help='Path to input configs file')
args = parser.parse_args()


def single_loop(configs: dict):
    """Trains a single model using a training and validation set"""

    # load tensorflow to initialize Cuda for subprocess (cross-val specifically)
    import tensorflow as tf

    # load dataset
    print(f"\nCreating dataset from `{configs['dataset_name']}`...\n")
    dataset = dataloader.create_train_dataset(configs)
    print(f"Training images: {os_sorted(configs['training_set'])}")
    print(f"Validation images: {os_sorted(configs['validation_set'])}\n")

    # load model
    print(f"Loading and compiling `{configs['encoder_name']}-{configs['decoder_name']}`...\n")
    model = models.load_UNet(configs)
    model.compile(
        optimizer = Adam(learning_rate=configs['learning_rate']), 
        loss = 'binary_crossentropy', 
        metrics = ['accuracy', 'Precision', 'Recall']
    )

    # save model summary if desired
    if configs['model_summary']:
        utils.save_model_summary(configs, model)

    # define training callbacks
    callbacks = [
        CSVLogger(
            str(configs['results_dir'] / 'metrics.csv'), 
            separator=',', 
            append=False
        ),
        ModelCheckpoint(
            str(configs['results_dir'] / 'best_model.keras'), 
            verbose=1, 
            save_best_only=True, 
            save_weights_only=False
        )
    ]
    if configs['early_stopping']:
        callbacks.append(EarlyStopping(patience = configs['patience']))
    
    # start training
    print("Training model...\n")
    history = model.fit(
        dataset['train_dataset'],
        epochs = configs['num_epochs'],
        steps_per_epoch = dataset['train_steps'],
        validation_data = dataset['val_dataset'],
        validation_steps = dataset['val_steps'],
        callbacks=callbacks,
        verbose=2
    )

    # load best model and inference train/val sets
    print("\nInferencing training and validation images with best model...\n")
    inf_model = load_model(str(configs['results_dir'] / 'best_model.keras'))
    inf_dataset = dataloader.create_train_val_inference_dataset(configs)
    inference(configs, inf_dataset, inf_model)

    # plot metrics
    print("\nPlotting metrics...\n")
    utils.plot_results(configs)

    print("Cleaning up...\n")
    # remove training dataset and clear memory
    shutil.rmtree(configs['root_dir'] / 'dataset')
    del model, inf_model, dataset, inf_dataset, history, callbacks
    K.clear_session()
    gc.collect()


def crossval_loop(configs: dict):
    """Trains num_folds models using different non-overlapping validation sets"""

    # determine all training and val set combinations given the number of folds
    train_sets, val_sets = utils.create_folds(configs['training_set'], configs['num_folds'])
    
    # save original results directory
    top_level_results = configs['results_dir']
    
    print(f"\nStarting cross validation with {configs['num_folds']} folds...\n")

    # loop through folds
    for fold in range(configs['num_folds']):
        print('-'*62 + ' Fold {} '.format(fold+1) + '-'*62)

        # update train/val sets
        configs.update({'training_set' : train_sets[fold]})
        configs.update({'validation_set' : val_sets[fold]})

        # create results directory for this fold
        results_dir = configs['results_dir'] / ('fold_' + str(fold+1))
        configs.update({'results_dir' : results_dir})
        results_dir.mkdir()

        # train fold as subprocess to prevent OOM
        p = multiprocessing.Process(target=single_loop, args=(configs,))
        p.start()
        p.join()

        # reset results directory for next fold
        configs.update({'results_dir' : top_level_results})

    # plot metrics over all folds
    print("Plotting cross validation results...\n")
    utils.cv_plot_results(configs)


def inference(configs: dict, dataset: dict, model: keras.Model):
    """Either inferences a training-validation pair of images, or just a single set of images"""

    if configs['operation_mode'] == 'train':
        # process train and val images
        train_preds = model.predict(dataset['train_dataset'], verbose=2)
        val_preds = model.predict(dataset['val_dataset'], verbose=2)

        # define and make output directories
        train_save_dir = configs['results_dir'] / 'train_preds'
        val_save_dir = configs['results_dir'] / 'val_preds'
        train_save_dir.mkdir()
        val_save_dir.mkdir()

        # define pred save file paths
        train_save_paths = [train_save_dir / (file.stem + '.png') for file in dataset['train_paths']]
        val_save_paths = [val_save_dir / (file.stem + '.png') for file in dataset['val_paths']]

        # save predictions
        utils.save_preds(train_preds, train_save_paths)
        utils.save_preds(val_preds, val_save_paths)

        del model, dataset, train_preds, val_preds, train_save_paths, val_save_paths

    else:
        # load model and dataset
        print("\nLoading data and model...\n")
        model = load_model(str(configs['root_dir'] / configs['model_path']))
        dataset = dataloader.create_inference_dataset(configs)

        # process dataset
        print("Generating model predictions...\n")
        preds = model.predict(dataset['dataset'], verbose=2)

        # define output directories and paths
        print("\nSaving model predictions...\n")
        save_dir = configs['results_dir'] / 'preds'
        save_dir.mkdir()
        save_paths = [save_dir / (file.stem + '.png') for file in dataset['data_paths']]

        # save predictions
        utils.save_preds(preds, save_paths)


def main():
    """Validates configs and instantiates operations class"""

    print(f"Loading and validating input configurations file `{Path(args.configs).name}`...\n")

    # load config dict
    with open(args.configs, 'r') as f:
        configs: dict = json.load(f)

    # validate and print configs
    configs = input_validator.validate(configs)
    utils.print_save_configs(configs.copy())

    # training or inference
    if configs['operation_mode'] == 'train':
        if configs['cross_validation']:
            crossval_loop(configs)
        else:
            single_loop(configs)
    else:
        inference(configs, None, None)
    
    print("Done.")


if __name__ == '__main__':
    main()