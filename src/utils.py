import warnings, os

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
        warnings.warn('No training images specified. Assuming all images in `{}` are to be used for training and validation'.format(ds_path))

    # if no val hold out percentage is set, use a default value
    if 'val_hold_out' not in configs['data']:
        warnings.warn('Validation hold out percentage out not specified. Defaulting to 40%. \
                       Set `val_hold_out` under `data` for a custom value')
        configs['data'].update({'val_hold_out' : 0.4})
    
    # ensure val_hold_out is a valid value
    if (type(configs['data']['val_hold_out'])) == float and (0 < configs['data']['val_hold_out'] < 1):
        raise ValueError('Specify the validation hold out percentage as a decimal between 0 and 1 (exclusive)')

    # auto_split is on, but a validation set is specified
    if (configs['data']['auto_split'] == True) and ('val' in configs['data']):
        raise InputContradiction('Auto split was turned on, but a validation set was provided. \
                                  Place all filenames under `train` instead.')
    


    return configs