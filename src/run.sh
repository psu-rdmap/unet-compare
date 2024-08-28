#!/bin/bash

#SBATCH --account=xvw5285_p_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4 
#SBATCH --mem=8GB
#SBATCH --gpus=1 
#SBATCH --time=20:00:00 
#SBATCH --output=last_run.out

module load anaconda
conda activate $PWD/..
export TF_CPP_MIN_LOG_LEVEL=3
#python3 train.py --dataset psu_dislocation/neutron_ds/dataset.dat --training psu_dislocation/training.dat
python3 train.py --dataset psu_dislocation/all_ds/dataset.dat --training psu_dislocation/training.dat
conda deactivate
