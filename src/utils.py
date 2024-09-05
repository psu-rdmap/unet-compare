import os
import warnings

class InputContradiction(Exception):
    pass

def check_input(configs : dict) -> dict:
    """
    Checks input configs for errors

    Parameters
    ----------
    configs : dict
        Input configs containing information for data loading, training, and results processing
    """
    
    # warn user that no training image filenames are provided
    if 'train' not in configs['data']:
        ds_path = os.path.join('data/', configs['data']['dataset_prefix'], 'images/')
        print('---- No training images specified. Assuming all images in `{}` are to be used for training and validation ----\n'.format(ds_path))

    # if no val hold out percentage is set, use a default value
    if 'val_hold_out' not in configs['data']:
        print('---- Validation hold out percentage not specified. Defaulting to 40%. Set `val_hold_out` under `data` for a custom value ----\n')
        configs['data'].update({'val_hold_out' : 0.4})
    
    # ensure val_hold_out is a float
    if (type(configs['data']['val_hold_out']) != float) and ((0 > configs['data']['val_hold_out']) or (1 > configs['data']['val_hold_out'])):
        raise ValueError('The validation hold out percentage must be represented as a decimal/float')
    
    # ensure val_hold_out is in the right bounds
    if (0 > configs['data']['val_hold_out']) or (1 < configs['data']['val_hold_out']):
        raise ValueError('The validation hold out percentage must be a decimal between 0 and 1')

    # auto_split is on, but a validation set is specified
    if (configs['data']['auto_split'] == True) and ('val' in configs['data']):
        raise InputContradiction('Auto split was turned on, but a validation set was provided. Place all filenames under `train` instead.')

    return configs
