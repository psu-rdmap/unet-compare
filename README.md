# Handling Input
Inputs were originally going to be set using dictionaries in a Python script, however, this requires additional functionality to process the input. Specifically, the dictionary object would need to be compiled and imported, rather than a simple decoding, as is the case for aomething like JSON. An alternative would be to pickle the dictionary objects and unpickle at runtime, however, this adds complexity to the user and the latent representation of the input data is not easily read due to its conversion into a binary stream. To avoid this, JSON will be used due to its simple decoding in Python, and dictionary-like data structure. It can also easily be compressed and visualized using text editors.

# Dataset
The datasets used in the paper can be found in the `data/` directory. Each dataset has the following structure:
```
name
├── images
│   ├── 1.ext
│   ├── 2.ext
│   └── ...
└── annotations
│   ├── 1.ext
│   ├── 2.ext
    └── ...
```
For now, images are associated with their corresponding annotations by giving them the same filenames. In the future, an alternative method will be implemented so that the user can explicitly define the relationship for each image/annotation instance.

## Custom dataset
A custom dataset can be used by following these steps:
1. Arrange data into the above directory structure
2. Create a symlink in the `data/` directory via `ln -s /path/to/custom/dataset/ ~/unet-compare/data/`
3. Change the `dataset_prefix` value in the configs file to the dataset name