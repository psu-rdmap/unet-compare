import os
# suppress warnings when tensorflow is imported
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYCARET_CUSTOM_LOGGING_LEVEL'] = 'CRITICAL'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import json
import loops
from utils import check_input
import tensorflow as tf
import time

# show if the gpu is available
print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# user input
parser = argparse.ArgumentParser(description='U-Net Training')
parser.add_argument('configs', type=str, help='Path to configs for training and dataset parameters')
args = parser.parse_args()

# load config dict and check input
with open(args.configs, 'r') as f:
    configs = json.load(f)
print('\nChecking user input...')
time.sleep(1.5)
print()
check_input(configs)

def main():
    # print input to user for confirmation
    print('-'*20 + ' User Input ' + '-'*20)
    for key, val in configs.items():
        print(key + ':', val)
    print('-'*52)

    # run the chosen training loop
    if configs['training_loop'] == 'Single':
        loops.default(configs)
    elif configs['training_loop'] == 'CrossVal':
        loops.cross_val(configs)
    else:
        raise ValueError('{} is not a valid training loop name'.format(configs['training_loop']))

if __name__ == '__main__':
    main()