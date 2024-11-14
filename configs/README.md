# Configs Overview

Config files are user-specified input parameters for training. These include training hyperparameters, dataset parameters, paths, and other settings. Input parameters are defined using keyword-argument pairs with the [JSON](https://www.json.org/json-en.html) format. Upon running the code, the input parameters will be checked and modified as necessary. At the end, the file will be dumped into the corresponding results directory for reference.

There are three types of run modes that can be performed:

- `Single()` - training a single model
- `CrossVal()` - training multiple models using [k-fold cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))
- `Inference()` - inferencing only using a provided model

All modes share a few general parameters, but the training modes (`Single`, `CrossVal`) have many additional parameters, with some shared between them and others unique to the mode. The overall structure of the input parameters can understood as follows:

```
General
|
├── Training
|   |
│   ├── Single
|   |
│   └── CrossVal
|
└── Inference
```

All parameters have a default setting, so the user is tasked with only updating parameters that are relevant to the specific run. The following sections will go over every input parameter, their definition, expected values, and the default setting.


# General Configs - `Single`, `Inference`, `CrossVal`

This section refers to the general input parameters shared by all modes.

- `training_mode`
    - **Definition**: type of training mode to run
    - **Expected values**: `Single`, `CrossVal`, `Inference`
    - **Default**: `Single`

- `dataset_prefix`
    - **Definition**: name of directory in `unet-compare/data/` corresponding to the training or inference data
    - **Expected values**: `string`
    - **Default**: `gb`


# Training Configs - `Single`, `CrossVal`

This section refers to the additional input parameters shared by `Single` and `CrossVal` training modes.

- `encoder_name`
    - **Definition**: name of encoder architecture for the desired U-Net
    - **Expected values**: `UNet`, `EfficientNetB7`
    - **Default**: `UNet`

- `decoder_name`
    - **Definition**: name of decoder architecture for the desired U-Net
    - **Expected values**: `UNet`, `UNet++`
    - **Default**: `UNet`

- `encoder_filters`
    - **Definition**: number of filters to use for each convolution operation for each row 
    - **Expected values**: `[int, int, int, int, int]`
    - **Default**: `[64, 128, 256, 512, 1024]`

- `decoder_filters`
    - **Definition**: number of filters to use for each convolution operation for each row 
    - **Expected values**: `[int, int, int, int, int]`
    - **Default**: `[512, 256, 128, 64, 32]`

- `freeze_backbone`
    - **Definition**: option for whether to freeze the pre-trained backbone. Only applies to the `EfficientNetB7` encoder
    - **Expected values**: `bool`
    - **Default**: `false`

- `image_ext`
    - **Definition**: filename extension for the training images
    - **Expected values**: `.jpg`, `.png`, etc.
    - **Default**: `.jpg`  

- `annotation_ext`
    - **Definition**: filename extension for the annotation images
    - **Expected values**: `.jpg`, `.png`, etc.
    - **Default**: `.png` 

- `learning_rate`
    - **Definition**: learning rate for the ADAM optimizer
    - **Expected values**: `float`
    - **Default**: `1e-4` 

- `l2_reg`
    - **Definition**: L2 regularization strength for convolution operations
    - **Expected values**: `float`
    - **Default**: `0.0` 

- `batch_size`
    - **Definition**: batch size to use for training
    - **Expected values**: `int`
    - **Default**: `1`

- `num_epochs`
    - **Definition**: maximum number of epochs to train for
    - **Expected values**: `int`
    - **Default**: `50` 

- `batchnorm` 
    - **Definition**: option for whether to use batch normalization layers or not
    - **Expected values**: `bool`
    - **Default**: `false` 

- `augment`
    - **Definition**: option for whether to augment the training dataset via flipping and rotations
    - **Expected values**: `bool`
    - **Default**: `true` 

- `save_best_only`
    - **Definition**: option for whether to checkpoint only when minimum validation loss has improved
    - **Expected values**: `bool`
    - **Default**: `true` 

- `standardize`
    - **Definition**: option for whether to [standardize](https://en.wikipedia.org/wiki/Standard_score) the dataset during loading
    - **Expected values**: `bool`
    - **Default**: `false` 


# `Single` Configs

This section refers to paremeters specific to the `Single` mode.

- `early_stopping`
    - **Definition**: option for whether to stop training if the validation loss has not improved after a set number of epochs determined by `patience`
    - **Expected values**: `bool`
    - **Default**: `true` 

- `patience`
    - **Definition**: number of epochs after which training will end if the validation loss has not improved
    - **Expected values**: `int`
    - **Default**: `10` 

- `train`
    - **Definition**: filenames (without the extension) of all image/annotation pairs to use for the training set. If provided, `val` will include all image/annotation pairs not specified. If not provided, it will include all image/annotation pairs not specified in `val`. If neither `train` nor `val` are provided, the data will be split automatically using all available data with a percentage specified in `val_hold_out` going to `val`
    - **Expected values**: `[string, string, ...]`
    - **Default**: `none`


- `val`
    - **Definition**: filenames (without the extension) of all image/annotation pairs to use for the validation set. If provided, `train` will include all image/annotation pairs not specified. If not provided, it will include all image/annotation pairs not specified in `train`. If neither `train` nor `val` are provided, the data will be split automatically using all available data with a percentage specified in `val_hold_out` going to `val`
    - **Expected values**: `[string, string, ...]`
    - **Default**: `none`

- `val_hold_out`
    - **Definition**: percentage of the dataset that will be used to make the validation dataset if neither `train` nor `val` are provided
    - **Expected values**: `float`
    - **Default**: `0.40`

- `checkpoint_path`
    - **Definition**: path to a previously trained checkpoint model to continue training
    - **Expected values**: `string`
    - **Default**: N/A (optional)


# `CrossVal` Configs

This section refers to paremeters specific to the `CrossVal` mode.

- `train`
    - **Definition**: filenames (without the extension) of all image/annotation pairs to use for cross-validation 
    - **Expected values**: `[string, string, ...]`
    - **Default**: all filenames found in `unet-compare/data/dataset_prefix/images/`

- `num_folds`
    - **Definition**: number of validation hold-out sets (folds) to train on
    - **Expected values**: `int`
    - **Default**: `3`


# `Inference` Configs

This section refers to additional input parameters specific to the `Inference` mode. The only other parameters that should be updated belong to the *General* class.
 
- `model_path`
    - **Definition**: path to a `.keras` model file relative to `/path/to/unet-compare`
    - **Expected values**: `string`
    - **Default**: must be provided


# Example Configs File
Below is an actual configs file used to train the baseline UNet model
