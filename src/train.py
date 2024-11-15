"""
This module handles all preliminary operations for training including taking input from the user, checking it, and calling the correct mode function
"""

import os
# suppress warnings when tensorflow is imported
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYCARET_CUSTOM_LOGGING_LEVEL'] = 'CRITICAL'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import json
import modes
import tensorflow as tf
from os import mkdir
from os.path import join
import checkers

# show if a gpu is available
print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# get config file from input
parser = argparse.ArgumentParser(description='U-Net Training')
parser.add_argument('configs', type=str, help='Path to configs for training and dataset parameters')
args = parser.parse_args()

# load config dict
with open(args.configs, 'r') as f:
    configs = json.load(f)

# perform general checks on user input
print('\nChecking user input...\n')
checkers.general(configs)


def main():  
    # perform mode-specific checks on user input then run the chosen mode
    if configs['training_mode'] == 'CrossVal':
        checkers.cross_val(configs)
        print_input(configs)
        modes.cross_val_mode(configs)

    elif configs['training_mode'] == 'Inference':
        checkers.inference(configs)
        print_input(configs)
        modes.inference_mode(configs)

    elif configs['training_mode'] == 'Single':
        checkers.single(configs)
        print_input(configs)
        modes.single_mode(configs)
    
    # save configs into results dir for reference
    with open(join(configs['results'], 'configs.json'), 'w') as con:
        json.dump(configs, con)


def print_input(configs : dict):
    # create top-level results directory
    mkdir(configs['results'])
    
    # print input to user for confirmation
    print('-'*50 + ' User Input ' + '-'*50)
    for key, val in configs.items():
        print(key + ':', val)
    print('-'*102)


if __name__ == '__main__':
    main()