"""
This module handles the definition of encoder/decoder subnetworks and their connection
"""

import keras
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Concatenate
from keras.regularizers import l2
from blocks import ConvBlock, UpsampleUnit
from tensorflow.keras.applications import EfficientNetB7


def UNet(configs : dict):
    enc_filters = configs['encoder_filters']
    dec_filters = configs['decoder_filters']
    batchnorm = configs['batchnorm']
    l2_reg = configs['l2_reg']

    # input
    input = keras.Input(shape = configs['input_shape'], name = 'main_input')

    # encoder
    conv_00 = ConvBlock(enc_filters[0], batchnorm, l2_reg, name = 'conv_block_00')(input)
    pool_00 = MaxPooling2D(pool_size = 2 , name = 'pool_00')(conv_00)

    conv_10 = ConvBlock(enc_filters[1], batchnorm, l2_reg, name = 'conv_block_10')(pool_00)
    pool_10 = MaxPooling2D(pool_size = 2 , name = 'pool_10')(conv_10)

    conv_20 = ConvBlock(enc_filters[2], batchnorm, l2_reg, name = 'conv_block_20')(pool_10)
    pool_20 = MaxPooling2D(pool_size = 2 , name = 'pool_20')(conv_20)

    conv_30 = ConvBlock(enc_filters[3], batchnorm, l2_reg, name = 'conv_block_30')(pool_20)
    pool_30 = MaxPooling2D(pool_size = 2 , name = 'pool_30')(conv_30)

    conv_40 = ConvBlock(enc_filters[4], batchnorm, l2_reg, name = 'conv_block_40')(pool_30)

    # decoder
    up_31 = UpsampleUnit(dec_filters[0], batchnorm, l2_reg, name='upsample_31')(conv_40)
    cat_31 = Concatenate(name='cat_31')([up_31, conv_30])
    conv_31 = ConvBlock(dec_filters[0], batchnorm, l2_reg, name = 'conv_block_31')(cat_31)

    up_22 = UpsampleUnit(dec_filters[1], batchnorm, l2_reg, name='upsample_22')(conv_31)
    cat_22 = Concatenate(name='cat_22')([up_22, conv_20])
    conv_22 = ConvBlock(dec_filters[1], batchnorm, l2_reg, name = 'conv_block_22')(cat_22)

    up_13 = UpsampleUnit(dec_filters[2], batchnorm, l2_reg, name='upsample_13')(conv_22)
    cat_13 = Concatenate(name='cat_13')([up_13, conv_10])
    conv_13 = ConvBlock(dec_filters[2], batchnorm, l2_reg, name = 'conv_block_13')(cat_13)

    up_04 = UpsampleUnit(dec_filters[3], batchnorm, l2_reg, name='upsample_04')(conv_13)
    cat_04 = Concatenate(name='cat_04')([up_04, conv_00])
    conv_04 = ConvBlock(dec_filters[3], batchnorm, l2_reg, name = 'conv_block_04')(cat_04)

    # final layer
    sigmoid = Conv2D(1, 1, activation='sigmoid', name='main_output', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2_reg))(conv_04)

    return keras.Model(inputs = input, outputs = sigmoid, name = 'UNet')


def EffUNet(configs : dict):
    enc_stages = ["stem_activation", "block2g_add", "block3g_add", "block5j_add", "block7d_add"]
    dec_filters = configs['decoder_filters']
    batchnorm = configs['batchnorm']
    l2_reg = configs['l2_reg']

    # input
    input = keras.Input(shape = configs['input_shape'], name = 'main_input')

    # encoder
    backbone = EfficientNetB7(
        include_top = False,
        weights = 'imagenet',
        input_tensor = input)

    # freeze entire backbone or just batchnorm layers
    if configs['freeze_backbone']:
        for layer in backbone.layers:
            layer.trainable = False
    else:
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    backbone_outputs = [backbone.get_layer(stage).output for stage in enc_stages]
    encoder = tf.keras.Model(inputs = input, outputs = backbone_outputs)
    enc_outputs = encoder.output

    # decoder
    up_31 = UpsampleUnit(dec_filters[0], batchnorm, l2_reg, name='upsample_31')(enc_outputs[4])
    cat_31 = Concatenate(name='cat_31')([up_31, enc_outputs[3]])
    conv_31 = ConvBlock(dec_filters[0], batchnorm, l2_reg, name = 'conv_block_31')(cat_31)

    up_22 = UpsampleUnit(dec_filters[1], batchnorm, l2_reg, name='upsample_22')(conv_31)
    cat_22 = Concatenate(name='cat_22')([up_22, enc_outputs[2]])
    conv_22 = ConvBlock(dec_filters[1], batchnorm, l2_reg, name = 'conv_block_22')(cat_22)

    up_13 = UpsampleUnit(dec_filters[2], batchnorm, l2_reg, name='upsample_13')(conv_22)
    cat_13 = Concatenate(name='cat_13')([up_13, enc_outputs[1]])
    conv_13 = ConvBlock(dec_filters[2], batchnorm, l2_reg, name = 'conv_block_13')(cat_13)

    up_04 = UpsampleUnit(dec_filters[3], batchnorm, l2_reg, name='upsample_04')(conv_13)
    cat_04 = Concatenate(name='cat_04')([up_04, enc_outputs[0]])
    conv_04 = ConvBlock(dec_filters[3], batchnorm, l2_reg, name = 'conv_block_04')(cat_04)

    # final layers
    up_final = UpsampleUnit(dec_filters[4], batchnorm, l2_reg, name = 'upsample_final')(conv_04)
    sigmoid = Conv2D(1, 1, activation='sigmoid', name='main_output', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2_reg))(up_final)

    return keras.Model(inputs=input, outputs=sigmoid, name = 'Eff-UNet')


def UNetPP(configs : dict):
    enc_filters = configs['encoder_filters']
    dec_filters = configs['decoder_filters']
    batchnorm = configs['batchnorm']
    l2_reg = configs['l2_reg']

    # input
    input = keras.Input(shape = configs['input_shape'], name = 'main_input')

    # encoder
    conv_00 = ConvBlock(enc_filters[0], batchnorm, l2_reg, name = 'conv_block_00')(input)
    pool_00 = MaxPooling2D(pool_size =2 , name = 'pool_00')(conv_00)

    conv_10 = ConvBlock(enc_filters[1], batchnorm, l2_reg, name = 'conv_block_10')(pool_00)
    pool_10 = MaxPooling2D(pool_size =2 , name = 'pool_10')(conv_10)

    conv_20 = ConvBlock(enc_filters[2], batchnorm, l2_reg, name = 'conv_block_20')(pool_10)
    pool_20 = MaxPooling2D(pool_size =2 , name = 'pool_20')(conv_20)

    conv_30 = ConvBlock(enc_filters[3], batchnorm, l2_reg, name = 'conv_block_30')(pool_20)
    pool_30 = MaxPooling2D(pool_size =2 , name = 'pool_30')(conv_30)

    conv_40 = ConvBlock(enc_filters[4], batchnorm, l2_reg, name = 'conv_block_40')(pool_30)

    # decoder
    up_31 = UpsampleUnit(dec_filters[0], batchnorm, l2_reg, name='upsample_31')(conv_40)
    cat_31 = Concatenate(name='cat_31')([up_31, conv_30])
    conv_31 = ConvBlock(dec_filters[0], batchnorm, l2_reg, name = 'conv_block_31')(cat_31)


    up_21 = UpsampleUnit(dec_filters[1], batchnorm, l2_reg, name='upsample_21')(conv_30)
    cat_21 = Concatenate(name='cat_21')([up_21, conv_20])
    conv_21 = ConvBlock(dec_filters[1], batchnorm, l2_reg, name = 'conv_block_21')(cat_21)

    up_22 = UpsampleUnit(dec_filters[1], batchnorm, l2_reg, name='upsample_22')(conv_31)
    cat_22 = Concatenate(name='cat_22')([up_22, conv_20, conv_21])
    conv_22 = ConvBlock(dec_filters[1], batchnorm, l2_reg, name = 'conv_block_22')(cat_22)


    up_11 = UpsampleUnit(dec_filters[2], batchnorm, l2_reg, name='upsample_11')(conv_20)
    cat_11 = Concatenate(name='cat_11')([up_11, conv_10])
    conv_11 = ConvBlock(dec_filters[2], batchnorm, l2_reg, name = 'conv_block_11')(cat_11)

    up_12 = UpsampleUnit(dec_filters[2], batchnorm, l2_reg, name='upsample_12')(conv_21)
    cat_12 = Concatenate(name='cat_12')([up_12, conv_10, conv_11])
    conv_12 = ConvBlock(dec_filters[2], batchnorm, l2_reg, name = 'conv_block_12')(cat_12)

    up_13 = UpsampleUnit(dec_filters[2], batchnorm, l2_reg, name='upsample_13')(conv_22)
    cat_13 = Concatenate(name='cat_13')([up_13, conv_10, conv_11, conv_12])
    conv_13 = ConvBlock(dec_filters[2], batchnorm, l2_reg, name = 'conv_block_13')(cat_13)


    up_01 = UpsampleUnit(dec_filters[3], batchnorm, l2_reg, name='upsample_01')(conv_10)
    cat_01 = Concatenate(name='cat_01')([up_01, conv_00])
    conv_01 = ConvBlock(dec_filters[3], batchnorm, l2_reg, name = 'conv_block_01')(cat_01)

    up_02 = UpsampleUnit(dec_filters[3], batchnorm, l2_reg, name='upsample_02')(conv_11)
    cat_02 = Concatenate(name='cat_02')([up_02, conv_00, conv_01])
    conv_02 = ConvBlock(dec_filters[3], batchnorm, l2_reg, name = 'conv_block_02')(cat_02)

    up_03 = UpsampleUnit(dec_filters[3], batchnorm, l2_reg, name='upsample_03')(conv_12)
    cat_03 = Concatenate(name='cat_03')([up_03, conv_00, conv_01, conv_02])
    conv_03 = ConvBlock(dec_filters[3], batchnorm, l2_reg, name = 'conv_block_03')(cat_03)

    up_04 = UpsampleUnit(dec_filters[3], batchnorm, l2_reg, name='upsample_04')(conv_13)
    cat_04 = Concatenate(name='cat_04')([up_04, conv_00, conv_01, conv_02, conv_03])
    conv_04 = ConvBlock(dec_filters[3], batchnorm, l2_reg, name = 'conv_block_04')(cat_04)

    # final layer
    sigmoid = Conv2D(1, 1, activation='sigmoid', name='main_output', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2_reg))(conv_04)

    return keras.Model(inputs = input, outputs = sigmoid, name = 'UNetpp')


def EffUNetPP(configs : dict):
    enc_stages = ["stem_activation", "block2g_add", "block3g_add", "block5j_add", "block7d_add"]
    dec_filters = configs['decoder_filters']
    batchnorm = configs['batchnorm']
    l2_reg = configs['l2_reg']

    # input
    input = keras.Input(shape = configs['input_shape'], name = 'main_input')

    # encoder
    backbone = EfficientNetB7(
        include_top = False,
        weights = 'imagenet',
        input_tensor = input)

    # freeze entire backbone or just batchnorm layers
    if configs['freeze_backbone']:
        for layer in backbone.layers:
            layer.trainable = False
    else:
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

    backbone_outputs = [backbone.get_layer(stage).output for stage in enc_stages]
    encoder = tf.keras.Model(inputs = input, outputs = backbone_outputs)
    enc_outputs = encoder.output

    # decoder
    up_31 = UpsampleUnit(dec_filters[0], batchnorm, l2_reg, name='upsample_31')(enc_outputs[4])
    cat_31 = Concatenate(name='cat_31')([up_31, enc_outputs[3]])
    conv_31 = ConvBlock(dec_filters[0], batchnorm, l2_reg, name = 'conv_block_31')(cat_31)


    up_21 = UpsampleUnit(dec_filters[1], batchnorm, l2_reg, name='upsample_21')(enc_outputs[3])
    cat_21 = Concatenate(name='cat_21')([up_21, enc_outputs[2]])
    conv_21 = ConvBlock(dec_filters[1], batchnorm, l2_reg, name = 'conv_block_21')(cat_21)

    up_22 = UpsampleUnit(dec_filters[1], batchnorm, l2_reg, name='upsample_22')(conv_31)
    cat_22 = Concatenate(name='cat_22')([up_22, enc_outputs[2], conv_21])
    conv_22 = ConvBlock(dec_filters[1], batchnorm, l2_reg, name = 'conv_block_22')(cat_22)


    up_11 = UpsampleUnit(dec_filters[2], batchnorm, l2_reg, name='upsample_11')(enc_outputs[2])
    cat_11 = Concatenate(name='cat_11')([up_11, enc_outputs[1]])
    conv_11 = ConvBlock(dec_filters[2], batchnorm, l2_reg, name = 'conv_block_11')(cat_11)

    up_12 = UpsampleUnit(dec_filters[2], batchnorm, l2_reg, name='upsample_12')(conv_21)
    cat_12 = Concatenate(name='cat_12')([up_12, enc_outputs[1], conv_11])
    conv_12 = ConvBlock(dec_filters[2], batchnorm, l2_reg, name = 'conv_block_12')(cat_12)

    up_13 = UpsampleUnit(dec_filters[2], batchnorm, l2_reg, name='upsample_13')(conv_22)
    cat_13 = Concatenate(name='cat_13')([up_13, enc_outputs[1], conv_11, conv_12])
    conv_13 = ConvBlock(dec_filters[2], batchnorm, l2_reg, name = 'conv_block_13')(cat_13)


    up_01 = UpsampleUnit(dec_filters[3], batchnorm, l2_reg, name='upsample_01')(enc_outputs[1])
    cat_01 = Concatenate(name='cat_01')([up_01, enc_outputs[0]])
    conv_01 = ConvBlock(dec_filters[3], batchnorm, l2_reg, name = 'conv_block_01')(cat_01)

    up_02 = UpsampleUnit(dec_filters[3], batchnorm, l2_reg, name='upsample_02')(conv_11)
    cat_02 = Concatenate(name='cat_02')([up_02, enc_outputs[0], conv_01])
    conv_02 = ConvBlock(dec_filters[3], batchnorm, l2_reg, name = 'conv_block_02')(cat_02)

    up_03 = UpsampleUnit(dec_filters[3], batchnorm, l2_reg, name='upsample_03')(conv_12)
    cat_03 = Concatenate(name='cat_03')([up_03, enc_outputs[0], conv_01, conv_02])
    conv_03 = ConvBlock(dec_filters[3], batchnorm, l2_reg, name = 'conv_block_03')(cat_03)

    up_04 = UpsampleUnit(dec_filters[3], batchnorm, l2_reg, name='upsample_04')(conv_13)
    cat_04 = Concatenate(name='cat_04')([up_04, enc_outputs[0], conv_01, conv_02, conv_03])
    conv_04 = ConvBlock(dec_filters[3], batchnorm, l2_reg, name = 'conv_block_04')(cat_04)

    # final layers
    up_final = UpsampleUnit(dec_filters[4], batchnorm, l2_reg, name = 'upsample_final')(conv_04)
    sigmoid = Conv2D(1, 1, activation='sigmoid', name='main_output', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(l2_reg))(up_final)

    return keras.Model(inputs=input, outputs=sigmoid, name = 'Eff-UNetpp')


def get_model(configs : dict):
    if configs['encoder_name'] == 'UNet':
        if configs['decoder_name'] == 'UNet':
            return UNet(configs)
        elif configs['decoder_name'] == 'UNet++':
            return UNetPP(configs)
    elif configs['encoder_name'] == 'EfficientNetB7':
        if configs['decoder_name'] == 'UNet':
            return EffUNet(configs)
        elif configs['decoder_name'] == 'UNet++':
            return EffUNetPP(configs)