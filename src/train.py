"""
This module handles all preliminary operations for training such as taking input from the user
"""

import os
# suppress warnings when tensorflow is imported
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYCARET_CUSTOM_LOGGING_LEVEL'] = 'CRITICAL'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import json
import loops
import tensorflow as tf
import time
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

# validate user input
print('\nChecking user input...\n')
time.sleep(2)
checkers.general(configs)

# print input to user for confirmation
print('-'*30 + ' User Input ' + '-'*30)
for key, val in configs.items():
    print(key + ':', val)
print('-'*72)

# create top-level results directory
mkdir(join(configs['root'], configs['results']))

# save configs into results dir for reference
with open(join(configs['root'], configs['results'], 'configs.json'), 'w') as con:
    json.dump(configs, con)

# run the chosen training loop
if configs['training_loop'] == 'CrossVal':
    checkers.cross_val(configs)
    loops.cross_val_loop(configs)
elif configs['training_loop'] == 'Inference':
    checkers.inference(configs)
    loops.inference_loop(configs)
elif configs['training_loop'] == 'Single':
    checkers.single(configs)
    loops.single_loop(configs)
