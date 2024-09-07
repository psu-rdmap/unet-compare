import os
import datetime

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
    

    # handle dataset prefix
    if 'dataset_prefix' not in configs:
        print('---- No dataset prefix provided. Defaulting to an empty string ----\n')
        configs.update({'dataset_prefix' : ''})
    else:
        assert type(configs['dataset_prefix']) == str, 'Dataset prefix must be a string'
    

    # check that an encoder is specified and is valid
    if 'encoder_name' not in configs:
        raise KeyError('No encoder name provided')
    else:
        encoders = ['Vanilla', 'EfficientNetB7']
        assert configs['encoder_name'] in encoders, 'Provided encoder name is invalid. Choose from: {}, {}'.format(*encoders)
  

    # ensure that the number of encoder filters is provided if training the UNet encoder
    if configs['encoder_name'] == 'Vanilla':
        assert 'encoder_filters' in configs, 'Encoder filter numbers must be provided if training the vanilla UNet encoder'
        assert type(configs['encoder_filters']) == list, 'Encoder filter numbers must be an array of 5 positive integers'
        assert len(configs['encoder_filters']) == 5, 'Encoder filter numbers must have 5 integers'
        assert all([type(filters) == int for filters in configs['encoder_filters']]), 'Encoder filter numbers must be integers'
        assert all([filters > 0 for filters in configs['encoder_filters']]), 'Encoder filter numbers must be positive'


    # add the decoder name to configs
    if 'decoder_name' not in configs:
        configs.update({'decoder_name' : 'UNet'})
    else:
        assert configs['decoder_name'] == 'UNet', 'Only the UNet decoder has been implemented'


    # ensure that the number of decoder filters is provided and correct
    assert 'decoder_filters' in configs, 'No decoder filter numbers provided. It must be a list of 4 integers'
    assert type(configs['decoder_filters']) == list, 'Decoder filter numbers must be an array of 4 positive integers'
    assert len(configs['decoder_filters']) == 4, 'Decoder filter numbers must have 4 integers'
    assert all([type(filters) == int for filters in configs['decoder_filters']]), 'Decoder filter numbers must be integers'
    assert all([filters > 0 for filters in configs['decoder_filters']]), 'Decoder filter numbers must be positive'


    # auto-detect root if it is not specified ahead of time or check the root thats been provided
    if 'root' not in configs:
        root = os.path.abspath(os.getcwd())
        configs.update({'root' : root})
        print('---- No root provided. Defaulting to the current working directory ({}) ----\n'.format(root))
    else:
        assert type(configs['root']) == str, 'Root directory must be a string'
        assert os.path.isdir(configs['root']), 'Root directory does not exist'

    
    # auto-define results directory if not provided or check the one thats been provided
    if 'results' in configs:
        print('---- Results directory provided. Overriding automatic definition ----\n')
        assert type(configs['results']) == str, 'Results directory must be a string'
        assert not os.path.isdir(configs['results']), 'Results directory already exists'
    else:
        now = datetime.datetime.now()
        results_dir = 'results_' + configs['dataset_prefix'] + '_' + configs['encoder_name'] + '_' + configs['decoder_name'] + now.strftime('_(%Y-%m-%d)_(%H-%M-%S)')
        configs.update({'results' : results_dir})
    

    # handle encoder stages
    if 'encoder_stages' not in configs:
        raise KeyError('No encoder stages array provided. Must be an array of 5 strings')
    else:
        assert type(configs['encoder_stages']) == list, 'Encoder stages must be an array of 5 strings'
        assert len(configs['encoder_stages']) == 5, 'Encoder stages must have 5 stages'
        assert all([type(stage) == str for stage in configs['encoder_stages']]), 'Encoder stages must be strings'


    # handle training loop
    if 'training_loop' not in configs:
        print('---- Training loop not provided. Defaulting to the single loop ----\n')
        configs.update({'training_loop' : 'Single'})
    else:
        training_loops = ['Single', 'CrossVal']
        assert configs['training_loop'] in training_loops, 'Provided training loop is invalid. Choose from: {}, {}'.format(*training_loops)


    # check learning rate
    if 'learning_rate' not in configs:
        raise KeyError('No learning rate provided')
    else:
        assert (type(configs['learning_rate']) == float) and (configs['learning_rate']  > 0), 'Learning rate must be a positive float'


    # set default value of l2 reg if not provided
    if 'l2_reg' not in configs:
        print('---- No L2 regularization strength provided. Defaulting to 0.0 ----\n'.format(root))
        configs.update({'l2_reg' : 0.0})
    else:
        assert type(configs['l2_reg']) == float and (configs['l2_reg']  > 0), 'L2 regularization strength must be a positive float'  


    # check batch size
    if 'batch_size' not in configs:
        raise KeyError('No batch size provided')
    else:
        assert (type(configs['batch_size']) == int) and (configs['batch_size']  > 0), 'Batch size must be a positive integer'


    # check number of epochs
    if 'num_epochs' not in configs:
        raise KeyError('No epoch number provided')
    else:
        assert (type(configs['num_epochs']) == int) and (configs['num_epochs']  > 0), 'Epoch number must be a positive integer'
   

    # set default batchnorm setting if not provided
    if 'batchnorm' not in configs:
        print('---- Batchnorm not provided. Defaulting to true ----\n'.format(root))
        configs.update({'batchnorm' : True})
    else:
        assert type(configs['batchnorm']) == bool, 'Batchnorm must be true or false'


    # only handle early stopping if it is provided
    if 'early_stopping' in configs:
        assert type(configs['early_stopping']) == bool, 'Early stopping must be true or false'
        if configs['early_stopping']:
            assert 'patience' in configs, 'Early stopping is true, but patience is not provided'
            assert type(configs['patience']) == int, 'Patience must be a positive integer'
            assert configs['patience'] > 0, 'Patience must be a positive integer'
    else:
        configs.update({'early_stopping' : False})


    # handle file extensions
    if 'image_ext' not in configs:
        raise KeyError('No image file extension provided')
    else:
        assert type(configs['image_ext']) == str, 'Image file extension must be a string'
        assert configs['image_ext'][0] == '.', 'Image file extension must start with a .'

    if 'annotation_ext' not in configs:
        raise KeyError('No annotation file extension provided')
    else:
        assert type(configs['annotation_ext']) == str, 'Annotation file extension must be a string'
        assert configs['annotation_ext'][0] == '.', 'Annotation file extension must start with a .'
    

    # handle input shape
    if 'input_shape' not in configs:
        raise KeyError('No desired input shape provided')
    else:
        assert len(configs['input_shape']) == 3, 'Input shape must follow [Height, Width, Channels]'
        assert all([type(shape) == int for shape in configs['input_shape']]), 'Input shape parameters must be positive integers'
        assert all([shape > 0 for shape in configs['input_shape']]), 'Input shape parameters must be positive integers'


    # handle augmentation
    if 'augment' not in configs:
        print('---- Augment not specified in data. Defaulting to false ----\n')
        configs.update({'augment' : False})
    else:
        assert type(configs['augment']) == bool, 'Augment must be true or false'


    # handle train (do not check for file existence; will be handled automatically in dataloader)
    if 'train' not in configs:
        ds_path = os.path.join('data/', configs['dataset_prefix'], 'images/')
        print('---- No training image filenames provided. Assuming all images in `{}` are to be used for training and validation ----\n'.format(ds_path))
    else:
        assert type(configs['train']) == list, 'Training image filenames must be an array of strings'
        assert len(configs['train']) > 0, 'At least one training image filename must be provided'
        assert all([type(fn) == str for fn in configs['train']]), 'All elements of the training filename set must be strings'


    # handle val 
    if 'val' not in configs:
        # check auto split
        if 'auto_split' not in configs:
            print('---- No validation set provided and neither was auto split. Defaulting latter to true ----\n')
            configs.update({'auto_split' : True})
        else:
            assert configs['auto_split'] == True, 'Auto split must be true if no validation set is provided'
    elif type(configs['val']) == list:
        assert (configs['auto_split'] == False) or ('auto_split' not in configs), 'Auto split must be false or not provided if no validation set is provided'
        assert len(configs['val']) > 0, 'At least one validation image filename must be provided if a validation set is provided'
        assert all([type(fn) == str for fn in configs['val']]), 'All elements of the validation filename set must be strings'
    else:
        raise KeyError('Validation set must be an array or not provided')
    

    # handle val hold out percentage
    if configs['auto_split'] == True:
        if 'val_hold_out' not in configs:
            raise KeyError('No validation hold out percentage provided, but auto split was true')
        elif 'val_hold_out' in configs:
            assert type(configs['val_hold_out'] == float), 'Validation hold out must be a decimal between 0 and 1'
            assert configs['val_hold_out'] > 0, 'Validation hold out must be a decimal between 0 and 1'
            assert configs['val_hold_out'] < 1, 'Validation hold out must be a decimal between 0 and 1'


    return configs