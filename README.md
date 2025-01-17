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

The architecture and operations are configured by the user using config files discussed in [`configs/README.md`](configs/README.md).

### Use with Linux/WSL
If you intend to use Linux or WSL2 with your own NVIDIA GPU, follow the installation instructions discussed [next](#installation). If using an AMD Radeon GPU, [ROCm](https://www.amd.com/en/products/software/rocm.html) will have to be used in place of CUDA. Note, a CPU can be used, however, training/inference takes an *immensely long time*. 

Support for multiple GPUs has not been implemented. If you want this functionality, modify the source code or submit an issue.


### Use with Google Colab
If you cannot use Linux or do not have a powerful GPU with sufficient memory, Google Colab is a good alternative. To use it, open the `unet_compare.ipynb` notebook and follow the instructions in the first cell.

You can use a lightweight GPU for free; more powerful GPUs are only available through the purchase of computational units. Priority access is enabled through a [subscription service](https://colab.research.google.com/signup).


# Installation
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
conda create -p env python 3.11
conda activate env/
```
Next, ensure the install of Python and pip (the package installer for Python) corresponds to the virtual environment.
```bash
which python   # /path/to/unet-compare/env/bin/python
which pip      # /path/to/unet-compare/env/bin/pip
```
If they do not point to the correct binaries, then the `PATH` shell variable must be updated with the correct binary location.
```bash
export PATH="/path/to/unet-compate/env/bin/<package>:$PATH"
```
The paths should be checked again to ensure the proper ones are used.

Dependencies can then be installed.
```bash
conda install cudatoolkit==11.8.0
pip install --upgrade pip
pip install -r requirements.txt
```
The project should now be installed. The following line checks which devices Tensorflow recognizes:
```bash
python scripts/check_gpu.py
```

# Training & Inference
This section gives information about running the source code. The first step is to create a dataset with the correct structure.

### 1. Creating a Dataset
Two training datasets are supplied in the `data/` directory. Each dataset has the following structure:
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
This format is for training and cross-validation. For inference, only the `images/` subdirectory is required.

Images are associated with their corresponding annotations by giving them the same filenames. All images or annotations must use a consistent file format. For example all images could be `.jpg` files, while all annotations could be `.png` files.

If you do not want to copy your own dataset into `data/`, it can be symlinked (i.e. create a shortcut) instead via:
```bash
ln -s /path/to/dataset/ /path/to/unet-compare/data/
```

### 2. Running `train.py`
The next step is to create a configs file in the `configs/` following the instructions in the [README](configs/README.md) file there. 

The script `train.py` is responsible for checking the configs and initiating the operation mode. Therefore, to start training, cross-validation, or inference, run the following line with your configs file:
```bash
python scr/train.py configs/<filename>.json
```
If you incorrectly defined any configs, an error will be thrown. Otherwise, operation will commence!

When you are finished, just deactivate the environment:
```bash
conda deactivate
```

### 3. Results
Results will be placed in the `results/` directory with a unique directory name describing the run.

If training (single and cross-validation), each results directory will contain the following
- Copy of the configs used
- Keras model file containing model structure and weights. This is what should be used for inference
- Model predictions for training and validation images
- CSV file with metrics calculated during training
- Plots of the loss and metrics
- Results from each fold (CV only)
- Statistical results (CV only)

If inferencing, there will instead be
- Copy of the configs used
- Model predictions

# Notes for Roar Collab Users (PSU)
If you are using Penn State's computing cluster, there are some additional considerations.

### Installation
When installing the project, Conda and Git tools will already be available. Also, you will have to use the command line via *Clusters* > *RC Shell Access* on the RC portal.

### Datasets
To upload data, you can use the RC portal's *Upload* function, an external tool like [*WinSCP*](https://winscp.net/eng/index.php), or command-line [file transfer methods](https://docs.icds.psu.edu/04_HandlingData/). 

### Running
To run `train.py`, a job must be submitted to the Slurm queue with a GPU-enabled account. There are two types of jobs: batch (`sbatch`) and interactive (`salloc`). Batch computing is an offline process, while interactive computing is tied to the runtime. 

**For batch jobs**, create a BASH script to handle resource acquisition and environment preparation filling in the details:
```bash
vim run.sh
# copy the following lines (with hashtags, they are Slurm directives), paste by right-clicking, and typing :wq

#!/bin/bash
#SBATCH --account=<account_name>
#SBATCH --nodes=1
#SBATCH --ntasks=4 
#SBATCH --mem=8GB
#SBATCH --gpus=1 
#SBATCH --time=<time>
#SBATCH --output=last_run.out
module load anaconda
conda activate $PWD/env
python src/train.py configs/<filename>.json
conda deactivate
```
Now you run the script:
```bash
sbatch run.sh
```
Once it is your turn in the queue, the shell script will be executed.

**For interactive jobs**, you acquire resources, activate the environment, do any work you need to do, and then exit:
```bash
salloc --account=<account_name> --ntasks=4 --gpus=1 --mem=32GB --time=1:00:00
conda activate unet-compare/env
# work ...
conda deactivate
exit
```

**Note:** you must be connected to a GPU-enabled account whenever running the code. Also, submitting an interactive job will automatically deactivate the virtual environment, so you must only activate after running `salloc`.

# References
[1] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” arXiv.org, May 18, 2015. https://arxiv.org/abs/1505.04597 <br/>
[2] Z. Zhou, M. Rahman, N. Tajbakhsh, and J. Liang, “UNet++: A Nested U-Net Architecture for Medical Image Segmentation,” arXiv.org, 2018. https://arxiv.org/abs/1807.10165 <br/>
[3] M. Tan and Q. V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” arXiv.org, 2019. https://arxiv.org/abs/1905.11946 <br/>