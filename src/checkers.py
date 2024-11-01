"""
This module handles all operations related to validating user inputs 
"""

import os
import datetime


def general(configs : dict):
    """
    Checks configs that are general to all loops

    Parameters
    ----------
    configs : dict
        Input configs defined in the JSON input file

    """
    
    # set root as absolute path to unet-compare
    current_dir = os.path.dirname(__file__)
    root = os.path.dirname(current_dir)
    configs.update({'root' : root})


    training_loops = ['Single', 'CrossVal', 'Inference']
    if 'training_loop' not in configs:
        raise KeyError('Training loop not provided. Choose from: {}, {}, {}'.format(*training_loops))
    else:
        assert configs['training_loop'] in training_loops, 'Provided training loop is invalid. Choose from: {}, {}, {}'.format(*training_loops)


    encoders = ['UNet', 'EfficientNetB7']
    if 'encoder_name' not in configs:
        raise KeyError('No encoder name provided. Choose from: {}, {}'.format(*encoders))
    else:
        assert configs['encoder_name'] in encoders, 'Provided encoder name is invalid. Choose from: {}, {}'.format(*encoders)

    
    decoders = ['UNet', 'UNet++']
    if 'decoder_name' not in configs:
        raise KeyError('Decoder name not provided. Choose from: {}, {}'.format(*decoders))
    else:
        assert configs['decoder_name'] in decoders, 'Provided decoder name is invalid. Choose from: {}, {}'.format(*decoders)


    # vanilla UNet encoder needs filter numbers
    if configs['encoder_name'] == 'Vanilla':
        assert 'encoder_filters' in configs, 'Encoder filter numbers must be provided if training the vanilla UNet encoder'
        assert type(configs['encoder_filters']) == list, 'Encoder filter numbers must be an array of 5 positive integers'
        assert len(configs['encoder_filters']) == 5, 'Encoder filter numbers must have 5 integers'
        assert all([type(filters) == int for filters in configs['encoder_filters']]), 'Encoder filter numbers must be integers'
        assert all([filters > 0 for filters in configs['encoder_filters']]), 'Encoder filter numbers must be positive'
    

    if 'decoder_filters' not in configs:
        KeyError('No decoder filter numbers provided. It must be a list of 4 integers')
    else:
        assert type(configs['decoder_filters']) == list, 'Decoder filter numbers must be an array of 5 positive integers'
        assert len(configs['decoder_filters']) == 5, 'Decoder filter numbers must have 4 integers'
        assert all([type(filters) == int for filters in configs['decoder_filters']]), 'Decoder filter numbers must be integers'
        assert all([filters > 0 for filters in configs['decoder_filters']]), 'Decoder filter numbers must be positive'


    if 'dataset_prefix' not in configs:
        raise KeyError('No dataset prefix provided. This is the dataset directory name found in data/')
    else:
        assert type(configs['dataset_prefix']) == str, 'Dataset prefix must be a string'
        data_path = os.path.join(configs['root'], 'data/', configs['dataset_prefix'])
        assert os.path.isdir(data_path), 'No directory exists at {}'.format(data_path)


    if 'input_shape' not in configs:
        raise KeyError('No input shape provided. It must be an array following: [Height, Width, Channels]')
    else:
        assert len(configs['input_shape']) == 3, 'Input shape must follow [Height, Width, Channels]'
        assert all([type(shape) == int for shape in configs['input_shape']]), 'Input shape parameters must be positive integers'
        assert all([shape > 0 for shape in configs['input_shape']]), 'Input shape parameters must be positive integers'


    if 'image_ext' not in configs:
        raise KeyError('No image file extension provided')
    else:
        assert type(configs['image_ext']) == str, 'Image file extension must be a string'
        assert configs['image_ext'][0] == '.', 'Image file extension must start with a .'


    now = datetime.datetime.now()
    results_dir = 'results_' + configs['dataset_prefix'] + '_' + configs['training_loop'] + '_' + configs['encoder_name'] + '_' + configs['decoder_name'] + now.strftime('_(%Y-%m-%d)_(%H-%M-%S)')
    results_dir = os.path.join(configs['root'], results_dir)
    configs.update({'results' : results_dir})


def training(configs : dict):
    """
    Checks configs that are specific to training loops single and cross validation 

    Parameters
    ----------
    configs : dict
        Input configs defined in the JSON input file with updates after general()

    """

    if 'learning_rate' not in configs:
        raise KeyError('No learning rate provided. It must be a float between 0 and 1')
    else:
        assert type(configs['learning_rate']) == float, 'Learning rate must be float between 0 and 1'     
        assert configs['learning_rate'] > 0, 'Learning rate must be float between 0 and 1'
        assert configs['learning_rate'] < 1, 'Learning rate must be float between 0 and 1'


    if 'l2_reg' not in configs:
        raise KeyError('No L2 regularization strength provided. It must be a float between 0 (inclusive) and 1')
    else:
        assert type(configs['l2_reg']) == float, 'L2 regularization must be a float between 0 (inclusive) and 1'
        assert configs['l2_reg'] >= 0, 'L2 regularization must be a float between 0 (inclusive) and 1'
        assert configs['l2_reg'] < 1, 'L2 regularization must be a float between 0 (inclusive) and 1'


    if 'batch_size' not in configs:
        raise KeyError('No batch size provided. It must be a positive integer')
    else:
        assert type(configs['batch_size']) == int, 'Batch size must be a positive integer'
        assert configs['batch_size'] > 0, 'Batch size must be a positive integer'


    if 'num_epochs' not in configs:
        raise KeyError('No number of epochs provided. It must be a positive integer')
    else:
        assert type(configs['num_epochs'] == int), 'Epoch number must be a positive integer'
        assert configs['num_epochs'] > 0, 'Epoch number must be a positive integer'
   

    if 'batchnorm' not in configs:
        raise KeyError('No batchnorm provided. It must be true or false')
    else:
        assert type(configs['batchnorm']) == bool, 'Batchnorm must be true or false'


    if 'annotation_ext' not in configs:
        raise KeyError('No annotation file extension provided')
    else:
        assert type(configs['annotation_ext']) == str, 'Annotation file extension must be a string'
        assert configs['annotation_ext'][0] == '.', 'Annotation file extension must start with a .'

    
    if 'augment' not in configs:
        raise KeyError('No augment provided. It must be true or false')
    else:
        assert type(configs['augment']) == bool, 'Augment must be true or false'


    if 'save_best_only' not in configs:
        raise KeyError('Specify whether to only checkpoint model when val loss improves. It must be true or false')
    else:
        assert type(configs['save_best_only']) == bool, 'Save best only must be true or false'


    if 'standardize' not in configs:
        raise KeyError('Specify whether to standardize the dataset prior to training. It must be true or false')
    else:
        assert type(configs['standardize']) == bool, 'Standardize must be true or false'


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
        raise KeyError('No early stopping provided. It must be true or false. Set to false if using cross validation')
    else:
        assert type(configs['early_stopping']) == bool, 'Early stopping must be true or false. Set to false if using cross validation'

        
    if configs['early_stopping']:
        if 'patience' not in configs:
            raise KeyError('Patience must be provided if using early stopping. It must be a positive integer')
        else:
            assert type(configs['patience']) == int, 'Patience must be a positive integer'
            assert configs['patience'] > 0, 'Patience must be a positive integer'


    if 'val' not in configs:
        if 'auto_split' not in configs:
            raise KeyError('Auto split must be true if no val set is provided')
        else:
            assert configs['auto_split'] == True, 'Auto split must be true if no val set is provided'
            if 'val_hold_out' not in configs:
                raise KeyError('A validation hold out percentage must be provided if no val set is provided')
            else:
                assert type(configs['val_hold_out'] == float), 'Validation hold out percentage must be a decimal between 0 and 1'
                assert configs['val_hold_out'] > 0, 'Validation hold out percentage must be a decimal between 0 and 1'
                assert configs['val_hold_out'] < 1, 'Validation hold out percentage must be a decimal between 0 and 1'
    else:
        assert type(configs['val']) == list, 'Validation image filenames must be an array of strings'
        assert configs['auto_split'] == False, 'Auto split must be false if a validation set is provided'
        assert len(configs['val']) > 0, 'At least one validation image filename must be provided if a validation set is provided'
        assert all([type(fn) == str for fn in configs['val']]), 'All elements of the validation filename set must be strings'


    if 'train' not in configs:
        data_path = os.path.join('data/', configs['dataset_prefix'], 'images/')
        assert os.path.isdir(data_path), 'Attempted to get retrieve filenames from {}, but the directory does not exist'.format(data_path)
        train_filenames = [os.path.splitext(fn)[0] for fn in os.listdir(data_path)]
        if 'val' not in configs:
            configs.update({'train' : train_filenames})
        else: 
            for val_fn in configs['val']: train_filenames.remove(val_fn)
            assert len(train_filenames) > 0, 'At least one image must be left for the train set '
            configs.update({'train' : train_filenames})
            assert ('auto_split' not in configs) or (configs['auto_split'] == False), 'Successfully retrieved train filenames, but auto split was true'            
    else:
        assert type(configs['train']) == list, 'Training image filenames must be an array of strings'
        assert len(configs['train']) > 0, 'At least one training image filename must be provided'
        assert all([type(fn) == str for fn in configs['train']]), 'All elements of the training filename set must be strings'

    
    if 'val' in configs:
        assert all([val_fn not in configs['train'] for val_fn in configs['val']]), 'Validation filename detected in training set'
            

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
        assert type(configs['train']) == list, 'Training image filenames must be an array of strings'
        assert len(configs['train']) > 0, 'At least one training image filename must be provided'
        assert all([type(fn) == str for fn in configs['train']]), 'All elements of the training filename set must be strings'


    if 'num_folds' not in configs:
        raise KeyError('No number of folds provided. It must be a positive integer greater than 1')
    else:
        assert type(configs['num_folds']) == int, 'Number of folds must be a positive integer greater than 1'
        assert configs['num_folds'] > 1, 'Number of folds must be a positive integer greater than 1'
        assert len(configs['train']) >= configs['num_folds'], 'There must be at least as many images as folds'


def inference(configs : dict):
    """
    Checks for weights/checkpoint file

    Parameters
    ----------
    configs : dict
        Input configs defined in the JSON input file with updates after general()

    Returns
    -------
    Updated configs : dict
    """

    if 'weights_path' not in configs:
        raise KeyError('No path to weights provided')
    else:
        assert type(configs['weights_path']) == str, 'Weights must be a path like string to the weights file'
        weights_path = os.path.join(configs['root'], configs['weights_path'])
        assert os.path.exists(weights_path), 'Weights file does not exist at {}'.format(weights_path) 
        configs['weights_path'] = weights_path 
