"""
This module handles all preliminary operations for training including taking input from the user, checking it, and calling the correct mode function
"""

import os
# suppress warnings when tensorflow is imported
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYCARET_CUSTOM_LOGGING_LEVEL'] = 'CRITICAL'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse, json, shutil, gc
import tensorflow as tf
from keras.api.optimizers import Adam
from keras.api.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.api.saving import load_model
from keras import backend as K
import input_validator, dataloader, models, utils

# show if a gpu is available
print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# get config file from input
parser = argparse.ArgumentParser(description='U-Net Training')
parser.add_argument('configs', type=str, help='Path to input configs file')
args = parser.parse_args()


class Operations:
    def __init__(self, configs: dict):
        self.configs = configs
        self.dataset = None
        self.model = None

    def single_loop(self):
        # load dataset and model
        self.dataset = dataloader_new.create_train_dataset(self.configs)
        self.model = models.load_UNet(self.configs)
        self.model.compile(
            optimizer = Adam(learning_rate=self.configs['learning_rate']), 
            loss = 'binary_crossentropy', 
            metrics = ['accuracy', 'Precision', 'Recall']
        )

        # define training callbacks
        callbacks = [
            CSVLogger(
                str(self.configs['results_dir'] / 'metrics.csv'), 
                separator=',', 
                append=False
            ),
            ModelCheckpoint(
                str(self.configs['results_dir'] / 'best_model.keras'), 
                verbose=1, 
                save_best_only=True, 
                save_weights_only=False
            )
        ]
        if self.configs['early_stopping']:
            callbacks.append(EarlyStopping(patience = self.configs['patience']))
        
        # start training
        self.model.fit(
            self.dataset['train_dataset'],
            epochs = self.configs['num_epochs'],
            steps_per_epoch = self.dataset['train_steps'],
            validation_data = self.dataset['val_dataset'],
            validation_steps = self.dataset['val_steps'],
            callbacks=callbacks,
            verbose=2
        )

        # load best model and inference train/val sets
        self.model = load_model(str(self.configs['results_dir'] / 'best_model.keras'))
        self.dataset = dataloader_new.create_train_val_inference_dataset(self.configs)
        self.inference()

        # plot metrics
        utils.plot_results(self.configs)

        # remove training dataset and clear memory
        shutil.rmtree(self.configs['root_dir'] / 'dataset')
        K.clear_session()
        gc.collect()


    def crossval_loop(self):
        # determine all training and val set combinations given the number of folds
        train_sets, val_sets = utils.create_folds(self.configs['training_set'], self.configs['num_folds'])
        
        # save original results directory
        top_level_results = self.configs['results_dir']
        
        # loop through folds
        for fold in range(self.configs['num_folds']):
            # update train/val sets
            self.configs.update({'training_set' : train_sets[fold]})
            self.configs.update({'validation_set' : val_sets[fold]})

            # create results directory for this fold
            results_dir = self.configs['results_dir'] / ('fold_' + str(fold+1))
            self.configs.update({'results_dir' : results_dir})
            results_dir.mkdir()

            # train fold
            self.single_loop()

            # reset results directory for next fold
            self.configs.update({'results_dir' : top_level_results})

        # plot metrics over all folds
        utils.cv_plot_results(self.configs)
        
        # remove previous val from configs
        self.configs.pop('validation_set')


    def inference(self):
        if self.configs['operation_mode'] == 'train':
            # process train and val images
            train_preds = self.model.predict(self.dataset['train_dataset'], verbose=2)
            val_preds = self.model.predict(self.dataset['val_dataset'], verbose=2)

            # define output directories
            train_save_dir = self.configs['results_dir'] / 'train_preds'
            val_save_dir = self.configs['results'] / 'val_preds'

            # define pred save file paths
            train_save_paths = [train_save_dir / (file.stem + '.png') for file in self.dataset['train_img_paths']]
            val_save_paths = [val_save_dir / (file.stem + '.png') for file in self.dataset['val_img_paths']]

            # save predictions
            utils.save_preds(train_preds, train_save_paths)
            utils.save_preds(val_preds, val_save_paths)

        else:
            # load model and dataset
            self.model = load_model(str(self.configs['root_dir'] / self.configs['model_path']))
            self.dataset = dataloader_new.create_inference_dataset(self.configs)

            # process dataset
            preds = self.model.predict(self.dataset['dataset'])

            # define output directories and paths
            save_dir = self.configs['results_dir'] / 'preds'
            save_paths = [save_dir / (file.stem + '.png') for file in self.dataset['data_paths']]

            # save predictions
            utils.save_preds(preds, save_paths)


def main():
    # load config dict
    with open(args.configs, 'r') as f:
        configs: dict = json.load(f)

    # validate and print configs
    configs = input_validator.validate(configs)
    utils.print_save_configs(configs.copy())

    # start operations
    operations = Operations(configs)

    # training or inference
    if configs['operation_mode'] == 'train':
        if configs['cross_validation']:
            operations.crossval_loop()
        else:
            operations.single_loop()
    else:
        operations.inference()

if __name__ == '__main__':
    main()