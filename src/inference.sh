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
python3 inference_test.py --model 'unet' --checkpoint_path 'bubble_unet/results_psu_bubble_512_unet_(2024-06-18)_(13-18-51)/best_model_unet.h5' --results 'bubble_test/' --images 'psu_bubble/test/'
conda deactivate
