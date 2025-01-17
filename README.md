# Overview
This project uses U-Net architectures [1-3] for *binary segmentation*, where a model is trained to map each pixel in an image to the range $[0,1]$. The output pixel values are probabilities of belonging to the feature class ($1$ is a feature, $0$ is a non-feature).

This project implements four types of configurable architectures:
- U-Net
- U-Net++
- U-Net w/ EfficientNetB7 backbone
- U-Net++ w/ EfficientNetB7 backbone

There are three modes of operation available:
- *Single* training loop generating a single trained model
- *Cross validation* statistical study generating multiple trained models
- *Inference* on an image set using a previously trained model

The user supplies an input file with various details discussed in [`configs/README.md`](configs/README.md), which defines the operation mode, data paths, neural network architecture, and training hyperparameters.

# Installation
## Linux
Conda is used to contain the project and install the dependencies. If you don't have Conda already installed, it can be installed using Miniconda:
```bash
cd ~
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# follow installation procedures
# restart shell to finish install
```
Now that Conda is installed, the repository is cloned and a virtual environment is created.
```bash
git clone https://github.com/psu-rdmap/unet-compare.git
cd unet-compare
conda config --add channels conda-forge
conda create -p env
conda activate env/
```
Next, dependencies are installed.
```bash
conda install python==3.11 cudatoolkit==11.8.0
pip install requirements.txt
```
It is assumed that you have a GPU-enabled device. To check, run the following code in Python (just run `python` using the command line):
```python

```

## Roar Collab (PSU)
If using Penn State's HPC *Roar Collab*, Conda should already be installed.

## Google Colab


# Running


# Training Datasets
Two TEM dataset are supplied in the `data/` directory. Each dataset has the following structure:
```
name
├── images
│   ├── 1.ext
│   ├── 2.ext
│   └── ...
└── annotations
    ├── 1.ext
    ├── 2.ext
    └── ...
```
Images are associated with their corresponding annotations by giving them the same filenames. All files and annotations must use a consistent file extension type 

## Custom dataset
A custom dataset can be used by following these steps:
1. Arrange data into the above directory structure
2. Create a symlink in the `data/` directory via `ln -s /path/to/custom/dataset/ unet-compare/data/`
3. Change the `dataset_prefix` value in the configs file to the dataset name


# References
[1] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” arXiv.org, May 18, 2015. https://arxiv.org/abs/1505.04597 <br/>
[2] Z. Zhou, M. Rahman, N. Tajbakhsh, and J. Liang, “UNet++: A Nested U-Net Architecture for Medical Image Segmentation,” arXiv.org, 2018. https://arxiv.org/abs/1807.10165 <br/>
[3] M. Tan and Q. V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” arXiv.org, 2019. https://arxiv.org/abs/1905.11946 <br/>