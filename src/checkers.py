"""
This module handles all operations related to validating user inputs 
"""

import os
import datetime


def general(configs : dict):
    """
    Checks configs that are general to all modes

    Parameters
    ----------
    configs : dict
        Input configs defined in the JSON input file

    """
    
    # set root as absolute path to unet-compare
    current_dir = os.path.dirname(__file__)
    root = os.path.dirname(current_dir)
    configs.update({'root' : root})


    training_modes = ['Single', 'CrossVal', 'Inference']
    if 'training_mode' not in configs:
        configs.update({'training_mode' : 'Single'})
    else:
        assert configs['training_mode'] in training_modes, 'Provided training_mode is invalid. Choose from: {}, {}, {}'.format(*training_modes)


    if 'dataset_prefix' not in configs:
        configs.update({'dataset_prefix' : 'gb'})
    else:
        assert type(configs['dataset_prefix']) == str, 'dataset_prefix must be a string'
        data_path = os.path.join(configs['root'], 'data/', configs['dataset_prefix'])
        assert os.path.isdir(data_path), 'No directory exists at {}'.format(data_path)


    now = datetime.datetime.now()
    results_dir = 'results_' + configs['dataset_prefix'] + '_' + configs['training_mode'] + '_' + configs['encoder_name'] + '_' + configs['decoder_name'] + now.strftime('_(%Y-%m-%d)_(%H-%M-%S)')
    results_dir = os.path.join(configs['root'], results_dir)
    configs.update({'results' : results_dir})


def training(configs : dict):
    """
    Checks configs that are specific to training modes single and cross validation 

    Parameters
    ----------
    configs : dict
        Input configs defined in the JSON input file with updates after general()

    """
    
    encoders = ['UNet', 'EfficientNetB7']
    if 'encoder_name' not in configs:
        configs.update({'encoder_name' : 'UNet'})
    else:
        assert configs['encoder_name'] in encoders, 'Provided encoder_name is invalid. Choose from: {}, {}'.format(*encoders)

    
    decoders = ['UNet', 'UNet++']
    if 'decoder_name' not in configs:
        configs.update({'decoder_name' : 'UNet'})
    else:
        assert configs['decoder_name'] in decoders, 'Provided decoder_name is invalid. Choose from: {}, {}'.format(*decoders)


    # vanilla UNet encoder needs filter numbers
    if 'encoder_filters' not in configs:
        configs.update({'encoder_filters' : [64, 128, 256, 512, 1024]})
    else:
        assert type(configs['encoder_filters']) == list, 'encoder_filters must be an array of 5 positive integers'
        assert len(configs['encoder_filters']) == 5, 'encoder_filters must have 5 integers'
        assert all([type(filters) == int for filters in configs['encoder_filters']]), 'encoder_filters must be integers'
        assert all([filters > 0 for filters in configs['encoder_filters']]), 'encoder_filters must be positive'
    

    if 'decoder_filters' not in configs:
        configs.update({'encoder_filters' : [512, 256, 128, 64, 32]})
    else:
        assert type(configs['decoder_filters']) == list, 'decoder_filter must be an array of 5 positive integers'
        assert len(configs['decoder_filters']) == 5, 'decoder_filter must have 4 integers'
        assert all([type(filters) == int for filters in configs['decoder_filters']]), 'decoder_filter must be integers'
        assert all([filters > 0 for filters in configs['decoder_filters']]), 'decoder_filter must be positive'


    if ' freeze_backbone' not in configs:
        configs.update({'freeze_backbone' : False})
    else:
        assert type(configs['batchnorm']) == bool, 'freeze_backbone must be true or false'


    if 'image_ext' not in configs:
        configs.update({'image_ext' : '.jpg'})
    else:
        assert type(configs['image_ext']) == str, 'image_ext must be a string'
        assert configs['image_ext'][0] == '.', 'image_ext must start with \'.\''


    if 'annotation_ext' not in configs:
        configs.update({'annotation_ext' : '.png'})
    else:
        assert type(configs['annotation_ext']) == str, 'annotation_ext must be a string'
        assert configs['annotation_ext'][0] == '.', 'annotation_ext must start with a .'
        

    if 'learning_rate' not in configs:
        configs.update({'learning_rate' : 1e-4})
    else:
        assert type(configs['learning_rate']) == float, 'learning_rate must be float between 0 and 1'     
        assert configs['learning_rate'] > 0, 'learning_rate must be float between 0 and 1'
        assert configs['learning_rate'] < 1, 'learning_rate must be float between 0 and 1'


    if 'l2_reg' not in configs:
        configs.update({'l2_reg' : 0.0})
    else:
        assert type(configs['l2_reg']) == float, 'l2_reg must be a float between 0 (inclusive) and 1'
        assert configs['l2_reg'] >= 0, 'l2_reg must be a float between 0 (inclusive) and 1'
        assert configs['l2_reg'] < 1, 'l2_reg must be a float between 0 (inclusive) and 1'


    if 'batch_size' not in configs:
        configs.update({'batch_size' : 1})
    else:
        assert type(configs['batch_size']) == int, 'batch_size must be a positive integer'
        assert configs['batch_size'] > 0, 'batch_size must be a positive integer'


    if 'num_epochs' not in configs:
        configs.update({'num_epochs' : 50})
    else:
        assert type(configs['num_epochs'] == int), 'num_epochs must be a positive integer'
        assert configs['num_epochs'] > 0, 'num_epochs must be a positive integer'
   

    if 'batchnorm' not in configs:
        configs.update({'batchnorm' : False})
    else:
        assert type(configs['batchnorm']) == bool, 'batchnorm must be true or false'

    
    if 'augment' not in configs:
        configs.update({'augment' : True})
    else:
        assert type(configs['augment']) == bool, 'augment must be true or false'


    if 'save_best_only' not in configs:
        configs.update({'save_best_only' : True})
    else:
        assert type(configs['save_best_only']) == bool, 'save_best_only must be true or false'


    if 'standardize' not in configs:
        configs.update({'standardize' : False})
    else:
        assert type(configs['standardize']) == bool, 'standardize must be true or false'


def single(configs : dict):
    """
    Checks early stopping and val settings

    Parameters
    ----------
    configs : dict
        Input configs defined in the JSON input file with updates after general() and training()

    """

    training(configs)
    

    if 'early_stopping' not in configs:
        configs.update({'early_stopping' : True})
    else:
        assert type(configs['early_stopping']) == bool, 'early_stopping must be true or false. Set to false if using cross validation'

        
    if configs['early_stopping']:
        if 'patience' not in configs:
            configs.update({'patience' : 10})
        else:
            assert type(configs['patience']) == int, 'patience must be a positive integer'
            assert configs['patience'] > 0, 'patience must be a positive integer'


    if 'train' not in configs:
        if 'val' not in configs: # train NO, val NO
            configs.update({'auto_split' : True})
            if 'val_hold_out' not in configs:
                raise KeyError('val_hold_out must be provided if no val set is provided. It refers to the percentage of the data to hold out for validation')
            else:
                assert type(configs['val_hold_out'] == float), 'val_hold_out must be a decimal between 0 and 1'
                assert configs['val_hold_out'] > 0, 'val_hold_out must be a decimal between 0 and 1'
                assert configs['val_hold_out'] < 1, 'val_hold_out must be a decimal between 0 and 1'
        else: # train NO, val YES
            assert type(configs['val']) == list, 'val must be an array of strings'
            assert configs['auto_split'] == False, 'auto_split must be false if a validation set is provided'
            assert len(configs['val']) > 0, 'At least one validation image filename must be provided if a validation set is provided'
            assert all([type(fn) == str for fn in configs['val']]), 'All elements of the validation filename set must be strings'            
            configs.update({'auto_split' : False})
            data_path = os.path.join(configs['root'], 'data/', configs['dataset_prefix'], 'images/')
            train_fns = [os.path.splitext(fn)[0] for fn in os.listdir(data_path)]
            for val_fn in configs['val']: train_fns.remove(val_fn)
            assert len(train_fns) > 0, 'At least one image must be left over for the training set'
            configs.update({'train' : train_fns})
    else:  
        configs.update({'auto_split' : False})
        assert type(configs['train']) == list, 'train must be an array of strings'
        assert len(configs['train']) > 0, 'At least one training image filename must be provided'
        assert all([type(fn) == str for fn in configs['train']]), 'All elements of the training filename set must be strings'
        if 'val' not in configs: # train YES, val NO
            data_path = os.path.join(configs['root'], 'data/', configs['dataset_prefix'], 'images/')
            val_fns = [os.path.splitext(fn)[0] for fn in os.listdir(data_path)]
            for train_fn in configs['train']: val_fns.remove(train_fn)
            assert len(val_fns) > 0, 'At least one image must be left over for the validation set'
            configs.update({'val' : val_fns})
        else: # train YES, val YES
            assert type(configs['val']) == list, 'val must be an array of strings'
            assert len(configs['val']) > 0, 'At least one validation image filename must be provided if a validation set is provided'
            assert all([type(fn) == str for fn in configs['val']]), 'All elements of the validation filename set must be strings'
            assert all([val_fn not in configs['train'] for val_fn in configs['val']]), 'Validation filename detected in training set'


    if 'checkpoint_path' in configs:
        assert type(configs['checkpoint_path']) == str, 'If continuing training from a previous model, checkpoint_path must be a path like string to the model .keras file. It should be relative to root'
        model_path = os.path.join(configs['root'], configs['checkpoint_path'])
        assert os.path.exists(model_path), 'Model file does not exist at {}'.format(model_path)
        configs['checkpoint_path'] = model_path
    else:
        configs.update({'checkpoint_path' : None})
            

def cross_val(configs : dict):
    """
    Checks val settings and for number of folds

    Parameters
    ----------
    configs : dict
        Input configs defined in the JSON input file with updates after general() and training()

    Returns
    -------
    Updated configs : dict
    """

    training(configs)

    configs.update({'early_stopping' : False})
    configs.update({'auto_split' : False})
    assert 'val' not in configs, 'Do not provide a validation set, if using cross-validation'

    
    if 'train' not in configs:
        data_path = os.path.join(configs['root'], 'data/', configs['dataset_prefix'], 'images/')
        assert os.path.isdir(data_path), 'Attempted to get retrieve filenames from {}, but the directory does not exist'.format(data_path)
        train_filenames = [os.path.splitext(fn)[0] for fn in os.listdir(data_path)]
        configs.update({'train' : train_filenames})
    else:
        assert type(configs['train']) == list, 'train must be an array of strings'
        assert len(configs['train']) > 0, 'At least one training image filename must be provided'
        assert all([type(fn) == str for fn in configs['train']]), 'All elements of the training filename set must be strings'


    if 'num_folds' not in configs:
        configs.update({'num_folds' : 3})
    else:
        assert type(configs['num_folds']) == int, 'num_folds must be a positive integer greater than 1'
        assert configs['num_folds'] > 1, 'num_folds must be a positive integer greater than 1'
        assert len(configs['train']) >= configs['num_folds'], 'There must be at least as many images as folds'


def inference(configs : dict):
    """
    Checks for model file

    Parameters
    ----------
    configs : dict
        Input configs defined in the JSON input file with updates after general()

    Returns
    -------
    Updated configs : dict
    """

    if 'model_path' not in configs:
        raise KeyError('model_path not provided. It must be a path like string to the model file relative to root')
    else:
        assert type(configs['model_path']) == str, 'model_path must be a path like string to the .keras model file. It should be relative to root'
        model_path = os.path.join(configs['root'], configs['model_path'])
        assert os.path.exists(model_path), 'Model file does not exist at {}'.format(model_path) 
        configs['model_path'] = model_path 