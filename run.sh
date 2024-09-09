#!/bin/bash

#SBATCH --account=xvw5285_p_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4 
#SBATCH --mem=8GB
#SBATCH --gpus=1 
#SBATCH --time=10:00:00 
#SBATCH --output=last_run.out

module load anaconda
conda activate $PWD/env
python src/train.py configs/demo.json
conda deactivate
