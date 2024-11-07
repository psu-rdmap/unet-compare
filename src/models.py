"""
This module handles the definition of encoder/decoder subnetworks and their connection
"""

import keras
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Concatenate
from keras.regularizers import l2
from blocks import ConvBlock, UpsampleBlock
from tensorflow.keras.applications import EfficientNetB7


def UNet(configs : dict):
    """
    U-Net built with Functional API using either U-Net or EfficientNetB7 encoders and either U-Net or U-Net++ decoders

    Parameters
    ----------
    configs : dict
        Input configs defined in the JSON input file

    Returns
    -------
    Keras Functional model representing the UNet : tf.keras.Model 
    """

    enc_filters = configs['encoder_filters']
    dec_filters = configs['decoder_filters']
    batchnorm = configs['batchnorm']
    l2_reg = configs['l2_reg']

    input = keras.Input(shape = configs['input_shape'], name = 'main_input')

    # encoder
    if configs['encoder_name'] == 'UNet':
        model_name = 'UNet'
        x = input
        enc_outputs = []
        for idx, filters in enumerate(enc_filters):
            name_idx = f'{idx}0'
            # conv
            x = ConvBlock(x, filters, batchnorm, l2_reg, name_idx)
            enc_outputs.append(x)
            # pool
            if idx < 4:
                x = MaxPooling2D(pool_size=2, name = 'pool_'+name_idx)(x)
        

    elif configs['encoder_name'] == 'EfficientNetB7':
        model_name = 'EfficientNet'
        backbone = EfficientNetB7(include_top = False, weights = 'imagenet', input_tensor = input)

        # freeze entire backbone or just batchnorm layers
        if configs['freeze_backbone']:
            for layer in backbone.layers:
                layer.trainable = False
        else:
            for layer in backbone.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False

        enc_stages = ['stem_activation', 'block2g_add', 'block3g_add', 'block5j_add', 'block7d_add']
        enc_outputs = [backbone.get_layer(stage).output for stage in enc_stages]

    
    # decoder
    enc_outputs.reverse()
    if configs['decoder_name'] == 'UNet':
        model_name += '-UNet'
        x = enc_outputs[0]
        for idx, filters in enumerate(dec_filters[:-1]):
            name_idx = f'{3-idx}{idx+1}' # 31, 22, 13, 04
            x = UpsampleBlock(x, dec_filters[idx], batchnorm, l2_reg, name_idx)
            x = Concatenate(name='cat_'+name_idx)([x, enc_outputs[idx+1]])
            x = ConvBlock(x, dec_filters[idx], batchnorm, l2_reg, name_idx)

    elif configs['decoder_name'] == 'UNet++':
        model_name += '-UNetpp'
        prev_row_outs = [enc_outputs[0]]
        for row in range(4): # number of rows
            current_row_outs = [enc_outputs[row+1]]
            for node in range(row+1): # 1, 2, 3, 4 nodes per row
                name_idx = f'{3-row}{node+1}' # 31, 21, 22, 11, 12, 13, 01, 02, 03, 04
                x = UpsampleBlock(prev_row_outs[node], dec_filters[row], batchnorm, l2_reg, name_idx)
                x = Concatenate(name='cat_'+name_idx)([x] + current_row_outs[:(node+1)])
                x = ConvBlock(x, dec_filters[row], batchnorm, l2_reg, name_idx)
                current_row_outs.append(x)
            prev_row_outs = current_row_outs

    # final layers
    if configs['encoder_name'] == 'EfficientNetB7':
        x = UpsampleBlock(x, dec_filters[4], batchnorm, l2_reg, 'final') # upsamples back to original resolution
    sigmoid = Conv2D(1, 1, activation='sigmoid', name='main_output', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2_reg))(x)

    return keras.Model(inputs=input, outputs=sigmoid, name = model_name)