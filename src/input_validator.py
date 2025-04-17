"""
Aiden Ochoa, 4/2025, RDMAP PSU Research Group
This module validates the user input configs file and modifies it as necessary
"""

from pydantic import BaseModel, PositiveInt, NonNegativeInt, PositiveFloat, NonNegativeFloat, ConfigDict, Field, field_validator, model_validator
from typing import Literal, List, Optional, Tuple
from pathlib import Path
from warnings import warn
import numpy as np
from natsort import os_sorted
from datetime import datetime
import cv2 as cv

ROOT_DIR = Path(__file__).parent.parent


class General(BaseModel):
    """Most general configs that apply to all possible use cases. Only basic validation is done here."""

    root_dir: Path = ROOT_DIR
    operation_mode: Literal['train', 'inference'] = Field(
        default='train',
        description="General parameter that defines whether a model will be trained, or if a model will be applied for inference"
    )
    dataset_name: str = Field(
        min_length=2,
        description="Dataset subdirectory prefix corresponding to unet-compare/data/<dataset_name>/"
    )
    model_config = ConfigDict(
        extra='allow',
    )


class Train(BaseModel):
    root_dir : Path
    operation_mode: str
    dataset_name: str
    input_shape: Tuple[int, int, int] = None
    encoder_name: Literal['UNet', 'EfficientNetB7'] = Field(
        default='UNet',
        description="Type of model architecture forming the encoder section of U-Net"
    )
    decoder_name: Literal['UNet', 'UNet++'] = Field(
        default='UNet',
        description="Type of model architecture forming the decoder section of U-Net"
    )
    encoder_filters: Optional[List[PositiveInt]] = Field(
        default=None,
        min_length=5, 
        max_length=5,
        description="Number of filters to learn for each convolution. Should be `null` or ignored when `encoder_name` is `EfficientNetB7`"
    )
    decoder_filters: List[PositiveInt] = Field(
        default = [512, 256, 128, 64, 32],
        description="Number of filters to learn at each resolution level. The final item is only used when `encoder_name` is `EfficientNetB7`"
    )
    backbone_weights: Optional[Literal['random', 'imagenet']] = Field(
        default=None,
        description="Weights to be loaded when using a pretrained backbone. This should be ignored when `encoder_name` is `UNet`"
    )
    backbone_finetuning: Optional[bool | List[NonNegativeInt]] = Field(
        default=None,
        max_length=7,
        description="Controls the finetuning of a pretrained backbone. " \
        "This should be ignored when random weights are used like when `encoder_name` is `UNet` and when `backbone_weights` is `random`." \
        "If the entire backbone is to be unfrozen, this should be `True`. Otherwise, `False` indicates the model is frozen or selected blocks (array of ints)"
    )
    learning_rate: PositiveFloat = Field(
        default=1e-4,
        lt=1.0,
        description="Learning rate for the Adam optimizer. Should be between 0 and 1"
    )
    L2_regularization_strength: NonNegativeFloat = Field(
        default=0.0,
        lt=1.0,
        description="Strength of L2 regularization used during training. Should be between 0 and 1, with 0 meaning no L2 regularization is used"
    )
    batch_size: PositiveInt = Field(
        default=4,
        description="Number of image-annotation pairs to use in a single batch. Each batch represents one weight vector update"
    )
    num_epochs: PositiveInt = Field(
        default=50,
        description="Number of epochs to train for. Each epoch is one pass through the entire dataset"
    )
    augment: bool = Field(
        default=True,
        description="Augment the training subset eightfold by flipping and rotating by 90 deg intervals"
    )
    save_model: bool = Field(
        default=True,
        description="Save the model to the results file"
    )
    standardize: bool = Field(
        default=False,
        description="Standardize the dataset using image statistics from the train subset"
    )
    cross_validation: bool = Field(
        default=False,
        description="Performs a k-fold cross validation study where many models are trained using different non-overlapping validation sets"
    )
    num_folds: Optional[PositiveInt] = Field(
        default=None,
        gt=1,
        description="Number of models to train for cross validation. Should be ignored when `cross_validation` is `false`"
    )
    early_stopping: Optional[bool] = Field(
        default=None,
        description="Stop the training if the validation loss does not improve after a given number of epochs provided by `patience`. " \
        "Does not apply when `cross_validation` is `true`"
    )
    patience: Optional[PositiveInt] = Field(
        default=None,
        description="Number of epochs before training is stopped automatically. Only applies when `early_stopping` is `true`"
    )
    training_set: Optional[List[str]] = Field(
        default=None,
        description="Array of image filenames to be used for the training set. Reference the logical tree to see how it may be defined"
    )
    validation_set: Optional[List[str]] = Field(
        default=None,
        description="Array of image filenames to be used for the validation set. Reference the logical tree to see how it may be defined"
    )
    auto_split: Optional[PositiveFloat] = Field(
        default=None, 
        lt=1,
        description="Validation hold out percentage used when automatically splitting the dataset. Reference the logical tree to see how it may be defined"
    )
    model_summary: bool = Field(
        default=True,
        description="Print out the model summary from Keras to a log file in the results directory"
    )
    results_dir: Optional[str | Path] = Field(
        default=None,
        description="Path for results directory relative to /path/to/unet-compare/. Give it `null` or ignore it to use default naming scheme"
    )
    batchnorm: bool = Field(
        default=False,
        description="Option to use batch normalization after convolution layers in the UNet encoder/decoder"
    )

    @model_validator(mode='after')
    def pretrained_backbone(self) -> 'Train':
        """Validation specific to pretrained backbones (EfficientNet)"""
        
        # set the encoder filters if not supplied and not using EfficientNet encoder
        if self.encoder_name == 'UNet' and self.encoder_filters is None:
            self.encoder_filters=[64, 128, 256, 512, 1024]
        
        # backbone weights should only be specified when using EfficientNet
        if self.encoder_name == 'UNet' and self.backbone_weights is not None:
            self.backbone_weights == None
            warn("`backbone_weights` should be `null` or ignored when `encoder_name` is `UNet`")
        elif self.encoder_name == 'EfficientNetB7' and self.backbone_weights is None: 
            raise ValueError(f"`backbone_weights` should be `random` or `imagenet` when `encoder_name` is `EfficientNetB7`")
        
        # backbone finetuning should be none when using random weights (UNet or random EfficientNet)
        if self.encoder_name == 'UNet' and self.backbone_finetuning is not None:
            self.backbone_finetuning = None
            warn("`backbone_finetuning` should be `null` or ignored when `encoder_name` is `UNet`")
        elif self.encoder_name == 'EfficientNetB7' and self.backbone_weights == 'random' and self.backbone_finetuning is not None:
            self.backbone_finetuning = None
            warn("`backbone_finetuning` should be `null` when `backbone_weights` is `random`")

        # block level unfreezing should be an array of block ints (0,1,2,...,7) 
        if self.encoder_name == 'EfficientNetB7' and type(self.backbone_finetuning) == list:
            assert len(set(self.backbone_finetuning)) == len(self.backbone_finetuning), "All block indices must be unique in `backbone_finetuning`"
            assert np.all(np.array(self.backbone_finetuning) < 8), "Block indices must be from 0 to 7 in `backbone_finetuning`"

        # default value is True (unfrozen backbone) if null
        if self.encoder_name == 'EfficientNetB7' and self.backbone_finetuning is None and self.backbone_weights == 'imagenet':
            warn(f"Expected `backbone_finetuning` to be an `array`, `true`, or `false`. Got `null` and defaulting to `True` (unfrozen)")
            self.backbone_finetuning = True

        return self

    @field_validator('dataset_name', mode='after')
    @classmethod
    def check_dataset(cls, dataset_name : str) -> str:
        """Checks various aspects about the dataset name provided"""
        
        # check if the dataset directory exists
        abs_path = ROOT_DIR / 'data' / dataset_name
        if not abs_path.exists():
            raise ValueError(f"Dataset can not be found at `{abs_path}`")

        # check if the dataset has the proper subdirectories
        img_subdir = abs_path / 'images'
        ann_subdir = abs_path / 'annotations'
        if not img_subdir.exists():
            raise ValueError(f"Dataset is missing the `images/` subdirectory")
        elif not ann_subdir.exists():
            raise ValueError(f"Dataset is missing the `annotations/` subdirectory")
        
        # check if the dataset subdirectories have no child directories themselves (only files)
        img_childdirs = [path.is_dir() for path in img_subdir.iterdir()]
        ann_childdirs = [path.is_dir() for path in ann_subdir.iterdir()]
        if any(img_childdirs):
            raise ValueError("`images/` subdirectory should contain files, not directories")
        elif any(ann_childdirs):
            raise ValueError("`annotations/` subdirectory should contain files, not directories")
            
        # check if the dataset subdirectories have at least 2 files
        img_files = list(img_subdir.iterdir())
        ann_files = list(ann_subdir.iterdir())
        num_imgs = len(img_files)
        num_anns = len(ann_files)
        if num_imgs != num_anns:
            raise ValueError(
                f"There must be the same number of images and annotations. Got {num_imgs} image files and {num_anns} annotation files")
        elif num_imgs < 2:
            raise ValueError("There must be at least 2 image/annotation file pairs")
        elif len(set(img_files)) != num_imgs: # set removes duplicates
            raise ValueError("Every image/annotation filename must be unique")
        
        # check if image/annotations have mixed file types 
        img_ext = {file.suffix for file in img_files}
        ann_ext = {file.suffix for file in ann_files}
        if len(img_ext) != 1:
            raise ValueError(f'Images must have the same file type. Got types {img_ext}')
        elif len(ann_ext) != 1:
            raise ValueError(f'Annotations must have the same file type. Got types {ann_ext}')
        
        # make sure image/annotations are are JPEGs or PNGs
        img_ext = next(iter(img_ext)) # gets element of singleton set
        ann_ext = next(iter(ann_ext))
        allowed_file_types = ['.jpg', '.jpeg', '.png']
        if img_ext not in allowed_file_types:
            raise ValueError(f'Expected image filetype to be one of {allowed_file_types}, got {img_ext}')
        if ann_ext not in allowed_file_types:
            raise ValueError(f'Expected annotation filetype to be one of {allowed_file_types}, got {ann_ext}')
               
        # check if every image has a corresponding annotation (by name)
        img_stems = {file.stem for file in img_files}
        ann_stems = {file.stem for file in ann_files}
        if img_stems != ann_stems:
            unpaired_stems = img_stems ^ ann_stems # get disjointed elements (unique to each set)
            raise ValueError(f'Found unpaired image or annotation files with filenames {unpaired_stems}')
        
        return dataset_name
    
    @field_validator('batch_size', mode='after')
    @classmethod
    def batch_size_warning(cls, batch_size : int) -> int:
        """Warn the user if batch size is not a power of 2"""
        if np.log2(batch_size) % 1 != 0.0:
            warn("`batch_size` is not a power of two. Efficiency may be reduced")
        
        return batch_size
    
    @model_validator(mode='after')
    def check_early_stopping(self) -> 'Train':
        """Checks early_stopping and patience fields and validate when doing cross validation"""
        
        # patience should only be provided when using early stopping and should be less than the number of epochs
        if self.early_stopping == False and self.patience is not None:
            self.patience = None
            warn(f"`patience` should be `null` if `early_stopping` is `false`. Changed `patience` to `null`")
        elif self.early_stopping == True:
            assert self.patience < self.num_epochs, "`patience` can not be greater than `num_epochs`"

        # early_stopping should be null when doing cross validation
        if self.cross_validation is True:
            if self.early_stopping is not None or self.patience is not None:
                self.early_stopping = None
                self.patience = None
                warn("`early_stopping` and `patience` should be `null` or ignored when `cross_validation` is `true`")

        return self

    @model_validator(mode='after')
    def check_train_val(self) -> 'Train':
        """Checks many aspects of the train-val splitting when training single models and doing cross validation. It applies the logical tree from the docs"""
        
        # dataset has already been validated since field_validators run first
        data_dir = ROOT_DIR / 'data' / self.dataset_name / 'images'
        img_stems = [file.stem for file in data_dir.iterdir()]
        img_stems_set = set(img_stems)

        # basic cross validation check 
        if self.cross_validation == True:
            # val sets will be determined algorithmically
            assert self.validation_set is None, "`validation_set` should be `null` or ignored when `cross_validation` is `true`"

        # logical tree with train at the top (see docs)

        # ------------ FORK 1: train set provided or not ------------ #
        if self.training_set is not None:
            # make sure all train files exist
            training_set = set(self.training_set)
            check_files(training_set, img_stems_set, 'train')

            # ------------ FORK 2: cross validation is true or not ------------ # 
            if self.cross_validation == True:
                # check number of folds
                assert self.num_folds < len(self.training_set), "`num_folds` can not be greater than the number of training images used for cross validation"

            else:
                # ------------ FORK 3: val set is provided or not ------------ # 
                if self.validation_set is not None:
                    # check overlap between train and val sets and make sure val set files exist
                    validation_set = set(self.validation_set)
                    train_val_overlap = training_set & validation_set
                    assert len(train_val_overlap) == 0, f"Files with the names {train_val_overlap} were found in both `training_set` and `validation_set`"
                    check_files(validation_set, img_stems_set, 'validation')
                else:

                    # ------------ FORK 4: auto split is provided or not ------------ # 
                    if self.auto_split:
                        # make sure auto_split is not too large where no train set is created
                        assert self.auto_split < (1 / len(self.training_set)), f"auto_split validation hold-out percentage must be less than {1/len(self.training_set)} for the `training_set` provided"
                    else:
                        # val set is the complement of train; make sure train does not have all available images
                        assert not training_set < img_stems_set, "`training_set` can not have all images in the dataset. Some must be left over for `validation_set`"
                        # generate val set (sort them naturally)
                        self.validation_set = os_sorted(list(img_stems_set - training_set))

        # ------------ FORK 1: back to the top (train set not provided) ------------ # 
        else:

            # ------------ FORK 2: cross validation is true or not ------------ # 
            if self.cross_validation == True:
                self.training_set = img_stems
            else:

                # ------------ FORK 3: val_set is provided or not ------------ #
                if self.validation_set is not None:
                    # train set is the complement to val set
                    validation_set = set(self.validation_set)
                    assert not validation_set < img_stems_set, "`validation_set` can not have all images in the dataset. Some must be left over for `training_set`"
                    # generate train set (sort them naturally)
                    self.training_set = os_sorted(list(img_stems_set - validation_set))
                else:
                    # define auto_split if it is not provided, or just check its value
                    if not self.auto_split:
                        self.auto_split = 0.4
                        warn("`auto_split` validation hold-out percentage not provided even though `training_set` and `validation_set` are `null` and `cross_validation` is `false`. Defaulting to 40%")
                    else:      
                        assert self.auto_split < 1 / len(img_stems_set), f"`auto_split` validation hold-out percentage must be less {1/len(img_stems_set)}"

        return self
    
    @model_validator(mode='after')
    def generate_results_dir(self) -> 'Train':
        """Create the results directory following a naming scheme if one is not provided"""

        if self.results_dir is None:
            now = datetime.now()
            self.results_dir = 'results_' + self.dataset_name + '_' + self.operation_mode + '_' + self.encoder_name + '_' + self.decoder_name 
            if self.cross_validation:
                self.results_dir += '_crossval'
            self.results_dir += now.strftime('_(%Y-%m-%d)_(%H-%M-%S)')
        
        self.results_dir = ROOT_DIR / self.results_dir

        return self
    
    @model_validator(mode='after')
    def get_image_shape(self) -> 'Train':
        """Get the image shape for model instantiation and make sure it is consistent"""
        
        data_dir = ROOT_DIR / 'data' / self.dataset_name

        # load each image or annotation and get the array shape; len(set)=1is found
        img_shapes = {cv.imread(str(img_path), cv.IMREAD_COLOR).shape for img_path in (data_dir / 'images').iterdir()}
        ann_shapes = {cv.imread(str(ann_path), cv.IMREAD_COLOR).shape for ann_path in (data_dir / 'annotations').iterdir()}
        img_shape = next(iter(img_shapes))
        ann_shape = next(iter(ann_shapes))

        # make sure all images and annotations have only one shape 
        if len(img_shapes) > 1:
            raise KeyError(f"Expected all images to have the same shape. Got shapes {img_shapes}")
        elif len(ann_shapes) > 1:
            raise KeyError(f"Expected all annotations to have the same shape. Got shapes {ann_shapes}")
        elif img_shape != ann_shape:
            raise KeyError(f"Expected images and annotations to have the same shape. Got an image shape of `{img_shape}` and an annotation shape of `{ann_shape}`")
        else:
            self.input_shape = img_shape

        return self

            
class Inference(BaseModel):
    root_dir: Path
    operation_mode: str
    dataset_name: str
    model_path: str = Field(
        description="Path relative to /path/to/unet-compare/ to an existing model to be used for inference"
    )
    results_dir: Optional[str] = Field(
        default=None,
        description="Path for results directory relative to /path/to/unet-compare/. Give it `null` or ignore it to use default naming scheme"
    )
    
    @field_validator('dataset_name', mode='after')
    @classmethod
    def check_dataset(cls, dataset_name : str) -> str:
        """Checks various aspects about the dataset name provided"""
        
        # check if the dataset directory exists
        abs_path = ROOT_DIR / 'data' / dataset_name
        if not abs_path.exists():
            raise ValueError(f"Dataset can not be found at `{abs_path}`")
        
        # check if the dataset directory has no child directories (only files)
        childdirs = [path.is_dir() for path in abs_path.iterdir()]
        if any(childdirs):
            raise ValueError("Dataset subdirectory should contain files, not directories")
            
        # check if the dataset has at least 1 file
        img_files = list(abs_path.iterdir())
        num_imgs = len(img_files)
        if not num_imgs:
            raise ValueError(f"There must be at least 1 image to inference in the dataset directory")
        
        # make sure images are JPEGs or PNGs
        img_exts = {file.suffix for file in img_files}
        allowed_file_types = ['.jpg', '.jpeg', '.png']
        unallowed_files = []
        for file in img_files:
            if file.suffix not in allowed_file_types:
                unallowed_files.append(file)
        if len(unallowed_files):
            raise ValueError(f'The files {unallowed_files} have invalid file types')
        
        # make sure images have the same file type
        if len(img_exts) > 1:
            raise KeyError(f"Expected all files to have the same file type. Got mixed types {img_exts}")
    
        return dataset_name

    @field_validator('model_path', mode='after')
    @classmethod
    def check_model(cls, model_path) -> 'General':
        # check if the model file exists
        abs_path = ROOT_DIR / model_path
        if not abs_path.exists():
            raise ValueError(f'No model file exists at `{abs_path}`')

        # check if it is a .keras model file
        ext = abs_path.name.split('.', 1)[-1]
        if ext == 'weights.h5':
            raise ValueError('Expected a .keras model file, got a weights.h5 file instead')
        elif ext == 'keras':
            pass
        else:
            raise ValueError(f'Expected a .keras file, got a .{ext} file')
        
        return model_path

    @model_validator(mode='after')
    def generate_results_dir(self) -> 'Train':
        """Create the results directory following a naming scheme if one is not provided"""

        if self.results_dir is None:
            now = datetime.now()
            self.results_dir = 'results_' + self.dataset_name + '_' + self.operation_mode + now.strftime('_(%Y-%m-%d)_(%H-%M-%S)')
        
        str(ROOT_DIR / self.results_dir)

        return self
    

def check_files(file_set: set, master_set: set, set_type: str):
    """Checks if all files in a set exist in a provided directory"""
    if not file_set <= master_set:
        missing_fns = file_set - master_set
        raise ValueError(f"Could not find files with the names {missing_fns} in the dataset given in `{set_type}_set")


def validate(input_configs: dict) -> dict:
    """Generate Pydantic models and validate input"""
    general = General.model_validate(input_configs)

    # select validator specific to the operation mode
    if general.operation_mode == 'train':
        output_configs = Train.model_validate(general.model_dump())
    else:
        output_configs = Inference.model_validate(general.model_dump())

    return output_configs.model_dump()