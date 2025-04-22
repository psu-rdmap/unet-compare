# Configs Overview

Config files contain are user-specified _configuration_ parameters which determine what happens during runtime. They can contain the operation mode (training or inference), dataset name, model hyperparameters, and so on. They are defined as keyword-argument pairs with the [JSON](https://www.json.org/json-en.html) format, and thus are expected to be `.json` files, or at the very least JSON serializable.

At the start of every run, the input parameters are validated according to predefined rules, and missing ones will be auto-populated with default values if possible. After validation, they are printed to the console for verification. You may notice extra parameters not mentioned below; these are system-generated and used throughout the runtime. Also, some parameters may be continually updated throughout the life of the run.

At the end of a run, the input parameters will be dumped into a file for later reference.

If nny bugs or issues are encountered with the input validation process, [submit an issue](https://github.com/psu-rdmap/unet-compare/issues) and it will discussed accordingly.


# Input Parameters

Every input file contains at the very least two configs

- `operation_mode`
    - **Description**: determines whether a model will be trained, or applied for inference
    - **Expected Values**: `train`, `inference`
    - **Default:**: `train`

- `dataset_name`
    - **Description**: name of dataset to be used for training, or to be inferenced, corresponding to the directory `/path/to/unet-compare/data/<dataset_name>`
    - **Expected Values**: `str`
    - **Default:**: `null` (required)

- `results_dir`
    - **Description**: name of results directory relative to `/path/to/unet-compare/`. If ignored (or `null`), the default naming scheme will be used which includes the operation mode, dataset name, cross validation, and a date-time stamp when the run was submitted
    - **Expected Values**: `str`
    - **Default:**: `null`


Depending on the selected operation mode, different input values will be expected.

## Training Configs

The following configs apply when `operation_mode` is `train` in the input file.

- `encoder_name`
    - **Definition**: name of encoder model for the desired U-Net architecture
    - **Expected values**: `UNet`, `EfficientNetB7`
    - **Default**: `UNet`

- `decoder_name`
    - **Definition**: name of decoder model for the desired U-Net architecture
    - **Expected values**: `UNet`, `UNet++`
    - **Default**: `UNet`

- `encoder_filters`
    - **Definition**: number of filters to learn for the convolution layers in each row of the encoder. This should be ignored (or `null`) when EfficientNetB7 is the encoder
    - **Expected values**: `[int, int, int, int, int]`
    - **Default**: `[64, 128, 256, 512, 1024]`

- `decoder_filters`
    - **Definition**: number of filters to learn for the convolution layers in each row of the decoder. The last item is only used when EfficientNetB7 is the encoder
    - **Expected values**: `[int, int, int, int, int]`
    - **Default**: `[512, 256, 128, 64, 32]`

- `batchnorm` 
    - **Definition**: adds a batch-normalization layer to each convolution layer in the U-Net encoder, U-Net decoder, and U-Net++ decoder
    - **Expected values**: `bool`
    - **Default**: `false` 

- `backbone_weights`
    - **Definition**: weights loaded when EfficientNetB7 is the encoder. This should be ignored (or `null`) when the default U-Net encoder is used
    - **Expected values**: `random`, `imagenet`
    - **Default**: `null` (required for EfficientNetB7)

- `backbone_finetuning`
    - **Definition**: controls how EfficientNetB7 is trained. This can be true to make the entire encoder trainable, false to make the entire encoder non-trainable, or specific block indices (`1` to `7`) can be provided in an array to make only them trainable. This should be ignored (or `null`) when the default U-Net encoder is used
    - **Expected values**: `bool`, `[1, 2, ..., 7]`
    - **Default**: `true`

- `learning_rate`
    - **Definition**: global learning rate $\eta$ for the Adam optimizer. This should be between `0.0` and `1.0`
    - **Expected values**: `float`
    - **Default**: `1e-4`

- `L2_regularization_strength`
    - **Definition**: strength of L2-regularization $\lambda$. This should be between `0.0` and `1.0`
    - **Expected values**: `float`
    - **Default**: `0.0`

- `batch_size`
    - **Definition**: number of image-annotation pairs to use in a single batch during training. The weight vector is updated via backpropagation after each batch is processed. This should be a power of 2 for computational efficiency
    - **Expected values**: `int`
    - **Default**: `4`

- `num_epochs`
    - **Definition**: number of epochs to train a model for. Each epoch is one pass through the entire training dataset, and is also how often the validation set is processed
    - **Expected values**: `int`
    - **Default**: `50`

- `augment`
    - **Definition**: augment the training set eightfold by flipping and rotating by 90 degrees
    - **Expected values**: `bool`
    - **Default**: `true`

- `cross_validation`
    - **Definition**: performs a k-fold cross validation study where many models are trained using different non-overlapping validation sets with the same hyperparameters
    - **Expected values**: `bool`
    - **Default**: `false`

- `num_folds`
    - **Definition**: number of non-overlapping validation sets to use with cross validation. It must be greater than `1` and less then the number of images in `training_set`. This should be ignored (or `null`) when `cross_validation` is `false`
    - **Expected values**: `int`
    - **Default**: `null`

- `early_stopping`
    - **Definition**: stop training a model early if the validation loss does not improve. This should be ignored (or `null`) when `cross_validation` is `true`
    - **Expected values**: `bool`
    - **Default**: `null`

- `patience`
    - **Definition**: number of epochs to wait after validation does not improve before training is stopped when `early_stopping` is `true`. This should be ignored (or `null`) when `cross_validation` is `true` 
    - **Expected values**: `int`
    - **Default**: `null` (required if `early_stopping` is `true`)

- `training_set`
    - **Definition**: filenames of image-annotation pairs to be used for training. This has complex logic associated with it and can be generated algorithmically if ignored or `null`. Look at this [flowchart]() to see how it may be specified
    - **Expected values**: `null`, `["fn_1", "fn_2", ...]`
    - **Default**: `null`

- `validation_set`
    - **Definition**: filenames of image-annotation pairs to be used for validation. This has complex logic associated with it and can be generated algorithmically if ignored or `null`. Look at this [flowchart]() to see how it may be specified. This should be ignored (or `null`) if `cross_validation` is `true`
    - **Expected values**: `null`, `["fn_1", "fn_2", ...]`
    - **Default**: `null`

- `auto_split`
    - **Definition**: percentage to hold-out from `training_set` for `validation_set`. If `training_set` is null, the percentage of all available image-annotation pairs specified by `dataset_name` will be held out. The upper limit is `1-1/len(training_set)`. This should only be specified if `cross_validation` is `false` and `validation_set` is `null`
    - **Expected values**: `int`
    - **Default**: `null` (`0.40` if `null` but required)

- `model_summary`
    - **Definition**: save a file showing the layers and parameter details, as well as a file showing which layers are trainable. The files will be saved to the results directory
    - **Expected values**: `bool`
    - **Default**: `true`


## Inference Configs

The following configs apply when `operation_mode` is `inference` in the input file.
 
- `model_path`
    - **Definition**: path to a `.keras` model file relative to `/path/to/unet-compare`
    - **Expected values**: `string`
    - **Default**: required


# Example Configs Files
Below is an example input file used to train the baseline U-Net model. It only includes parameters that differ from their default values. Following the logic in this [flowchart](), `training_set` will use all remaining image-annotation pairs in the dataset `data/gb_512/`. 

```JSON
{
    "L2_regularization_strength" : 1e-3,
    "num_epochs" : 500,
    "patience" : 50,
    "dataset_name" : "gb_512",
    "batch_size" : 2,
    "validation_set" : ["11_4", "30_1", "12_1", "1_3", "26_4", "16_2", "4_1", "1_1", "7_3", "25_4", "30_3", "16_3", "14_1", "30_2"]
}
```

Below is another example input file used to train a U-Net++ model with the EfficientNetB7 encoder with only its final three blocks being trainable. This demonstrates a case where `training_set` and `validation_set` are ignored. They they will be populated using 70% of the images-annotation pairs in `data/bubble_1024` for training, and 30% for validation.

```JSON
{
    "encoder_name" : "EfficientNetB7",
    "decoder_name" : "UNet++",
    "backbone_weights" : "imagenet",
    "backbone_finetuning" : [5, 6, 7],
    "l2_reg" : 1e-3,
    "num_epochs" : 50,
    "patience" : 10,
    "dataset_prefix" : "bubble_1024",
    "batch_size" : 4,
    "auto_split" : 0.30
}
```

The next example showcases 6-fold cross validation with a U-Net++ architecture. Here, the minimum number of image/annotation pairs given six folds are provided, thus being a special type of cross-validation known as *Leave One Out Cross Validation* (LOOCV). If `training_set` was null, all images in `data/bubble_1024` would be used. 

```JSON
{
    "decoder_name" : "UNet++",
    "training_mode" : "CrossVal",
    "L2_regularization_strength" : 1e-4,
    "num_epochs" : 50,
    "dataset_name" : "bubble_1024",
    "batch_size" : 4,
    "num_folds" : 6,
    "training_set" : ["1", "3", "5", "6", "9", "10"]
}
```

The final example is using a model that has been previously trained and applied to a set of hypothetical images.

```JSON
{
    "operation_mode" : "inference",
    "dataset_prefix" : "test_images",
    "model_path" : "results_gb_512_train_EfficientNetB7_UNet_(2025-04-18)_(14-33-35)/best_model.keras"
}
```