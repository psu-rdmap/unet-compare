"""
Aiden Ochoa, 4/2025, RDMAP PSU Research Group
This module handles all training and inference operations. It has __main__
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
from pathlib import Path
from natsort import os_sorted

# show if a gpu is available
print("\nNumber of GPUs available: ", len(tf.config.list_physical_devices('GPU')), '\n')

# get config file from input
parser = argparse.ArgumentParser(description='U-Net Training')
parser.add_argument('configs', type=str, help='Path to input configs file')
args = parser.parse_args()


class Operations:
    """Singleton class for training and inference"""

    def __init__(self, configs: dict):
        """Initialize configs, dataset, and model"""
        self.configs = configs
        self.dataset = None
        self.model = None

    def single_loop(self):
        """Trains a single model using a training and validation set"""

        # load dataset and model
        print(f"\nCreating dataset from `{self.configs['dataset_name']}`...\n")
        self.dataset = dataloader.create_train_dataset(self.configs)
        print(f"Training images: {os_sorted(self.configs['training_set'])}")
        print(f"Validation images: {os_sorted(self.configs['validation_set'])}\n")
        print(f"Loading and compiling `{self.configs['encoder_name']}-{self.configs['decoder_name']}`...\n")
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
        print("Training model...\n")
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
        print("\nInferencing training and validation images with best model...\n")
        self.model = load_model(str(self.configs['results_dir'] / 'best_model.keras'))
        self.dataset = dataloader.create_train_val_inference_dataset(self.configs)
        self.inference()

        # plot metrics
        print("\nPlotting metrics...\n")
        utils.plot_results(self.configs)

        print("Cleaning up...\n")
        # remove training dataset and clear memory
        shutil.rmtree(self.configs['root_dir'] / 'dataset')
        K.clear_session()
        gc.collect()


    def crossval_loop(self):
        """Trains num_folds models using different non-overlapping validation sets"""

        # determine all training and val set combinations given the number of folds
        train_sets, val_sets = utils.create_folds(self.configs['training_set'], self.configs['num_folds'])
        
        # save original results directory
        top_level_results = self.configs['results_dir']
        
        print(f"\nStarting cross validation with {self.configs['num_folds']} folds...\n")

        # loop through folds
        for fold in range(self.configs['num_folds']):
            print('-'*62 + ' Fold {} '.format(fold+1) + '-'*62)

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
        print("Plotting cross validation results...\n")
        utils.cv_plot_results(self.configs)


    def inference(self):
        """Either inferences a training-validation pair of images, or just a single set of images"""

        if self.configs['operation_mode'] == 'train':
            # process train and val images
            train_preds = self.model.predict(self.dataset['train_dataset'], verbose=2)
            val_preds = self.model.predict(self.dataset['val_dataset'], verbose=2)

            # define and make output directories
            train_save_dir = self.configs['results_dir'] / 'train_preds'
            val_save_dir = self.configs['results_dir'] / 'val_preds'
            train_save_dir.mkdir()
            val_save_dir.mkdir()

            # define pred save file paths
            train_save_paths = [train_save_dir / (file.stem + '.png') for file in self.dataset['train_paths']]
            val_save_paths = [val_save_dir / (file.stem + '.png') for file in self.dataset['val_paths']]

            # save predictions
            utils.save_preds(train_preds, train_save_paths)
            utils.save_preds(val_preds, val_save_paths)

        else:
            # load model and dataset
            print("\nLoading data and model...\n")
            self.model = load_model(str(self.configs['root_dir'] / self.configs['model_path']))
            self.dataset = dataloader.create_inference_dataset(self.configs)

            # process dataset
            print("Generating model predictions...\n")
            preds = self.model.predict(self.dataset['dataset'], verbose=2)

            # define output directories and paths
            print("\nSaving model predictions...\n")
            save_dir = self.configs['results_dir'] / 'preds'
            save_dir.mkdir()
            save_paths = [save_dir / (file.stem + '.png') for file in self.dataset['data_paths']]

            # save predictions
            utils.save_preds(preds, save_paths)


    def save_model_summary(self):
        """Writes the model summary to a file after training and writes a file about the trainable layers"""

        with open(self.configs['results_dir'] / 'model_summary.out', 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        with open(self.configs['results_dir'] / 'trainable.out', 'w') as f:
            f.write(f"{'Layer':<35} {'Trainable':<20}\n")
            f.write("=" * 50 + "\n")
            for layer in self.model.layers:
                f.write(f"{layer.name:<35} {str(layer.trainable):<20}\n")


def main():
    """Validates configs and instantiates operations class"""

    print(f"Loading and validating input configurations file `{Path(args.configs).name}`...\n")

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

        if configs['model_summary']:
            operations.save_model_summary()
    else:
        operations.inference()
    
    print("Done.")


if __name__ == '__main__':
    main()