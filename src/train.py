"""
This module handles all preliminary operations for training including taking input from the user, checking it, and calling the correct mode function
"""

import os
# suppress warnings when tensorflow is imported
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYCARET_CUSTOM_LOGGING_LEVEL'] = 'CRITICAL'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse, json, pathlib, shutil, gc
import tensorflow as tf
from keras.api.optimizers import Adam
from keras.api.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.api.saving import load_model
from keras import backend as K
import input_validator, dataloader_new, models

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
                str(self.configs['results'] / 'metrics.csv'), 
                separator=',', 
                append=False
            ),
            ModelCheckpoint(
                str(self.configs['results'] / 'best_model.keras'), 
                verbose=1, 
                save_best_only=True, 
                save_weights_only=False
            )
        ]
        if self.configs['early_stopping']:
            callbacks.append(EarlyStopping(patience = self.configs['patience']))
        
        # start training
        self.model.fit(
            self.dataset['train'],
            epochs = self.configs['num_epochs'],
            steps_per_epoch = self.dataset['train_steps'],
            validation_data = self.dataset['val'],
            validation_steps = self.dataset['val_steps'],
            callbacks=callbacks,
            verbose=2
        )

        # load best model and inference train/val sets
        self.model = load_model(str(self.configs['results'] / 'best_model.keras'))
        self.inference()

        # plot metrics


        # remove training dataset and clear memory
        shutil.rmtree(self.configs['root'] / 'dataset')
        K.clear_session()
        gc.collect()

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

    # inferences train/val sets and save results into results
    print('\nInferencing image sets...')
    utils.inference_ds(configs, model)

    # plot loss, precision, recall, and f1
    print('\nPlotting metrics...')
    utils.plot_results(configs)

    def crossval_loop(self):
        pass

    def inference(self):
        if self.configs['operation_mode'] == 'train':
            # get tensors of just train images and val images from self.dataset
            
            # unbatch and batch with a batch size of 1

            # model predict

            # 

        else:
            self.dataset = dataloader.create_inference_dataset(self.configs)


def main():
    # load config dict
    with open(args.configs, 'r') as f:
        configs = json.load(f)

    # validate configs
    print('\nValidating User Input...')
    configs = input_validator.validate(configs)
    
    # start operations
    operations = Operations(configs)

    # training or inference
    if configs['operation_mode'] == 'train':
        if configs['cross_val']:
            operations.crossval_loop()
        else:
            operations.single_loop()
    else:
        operations.inference()

    
    
    # save configs into results dir for reference
    with open(join(configs['results'], 'configs.json'), 'w') as con:
        json.dump(configs, con)


def print_configs(configs : dict):
    # create top-level results directory
    mkdir(configs['results'])
    
    # print input to user for confirmation
    print('-'*50 + ' User Input ' + '-'*75)
    for key, val in configs.items():
        print(key + ':', val)
    print('-'*137)


        

def cross_val_mode(configs : dict):    
    """
    Runs single training mode many times on different hold out sets

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
        single_mode(configs)

        # reset results directory for next fold
        configs.update({'results' : top_level_results})

    # plot metrics over all folds
    print('\nPlotting CV Metrics...')
    utils.cv_plot_results(configs)
    
    # remove previous val from configs
    configs.pop('val')

    print('\nDone.')


def inference_mode(configs : dict):
    """
    Feeds images into a trained model and saves predictions

    Parameters
    ----------
    configs : dict
        Input configs provided by the user
    """

    print('Loading images...')

    # define path to images and retrieve filenames
    data_path = join(configs['root'], 'data/', configs['dataset_prefix'])
    ds_fns = listdir(data_path)

    # create tensorflow ds
    ds = [join(data_path, fn) for fn in ds_fns]
    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = ds.map(utils.parse_inference_image)

    # build model and load weights
    print('\nLoading model...')
    model = keras.saving.load_model(configs['model_path'])

    # create and save predictions
    print('\nGenerating predictions and saving them...\n')
    preds = model.predict(ds, verbose=2)
    save_fns = [join(configs['results'], split(fn)[0], '.png') for fn in ds_fns]
    utils.save_preds(preds, save_fns)

    print('\nDone.')








if __name__ == '__main__':
    main()