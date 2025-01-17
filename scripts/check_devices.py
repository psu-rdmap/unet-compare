import os
# suppress warnings when tensorflow is imported
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def check_device(device_type : str):
    devices = tf.config.list_physical_devices(device_type)
    
    if devices:
        print(f'{device_type}s available: {len(devices)}')
        for i, device in enumerate(devices):
            print(f'{device_type} {i}: {device}')
    else:
        print(f'No {device_type}s available')

if __name__ == "__main__":
    check_device('CPU')
    print()
    check_device('GPU')
