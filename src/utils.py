import pickle

import sys
sys.path.append('/home/aidenochoa1/unet-compare/')
from configs.default import train_configs, data_configs

def test_input(configs):
    """
    Checks input configs for errors.

    Parameters
    ----------
    configs : dict
        input configs containing training information
    """
