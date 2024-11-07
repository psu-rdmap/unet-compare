"""
This module handles the definition of the custom convolution layers used in UNet models
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose
from keras.regularizers import l2


def ConvBlock(inputs, filters, batchnorm, l2_reg, index):
    """
    Two Conv2D with optional batchnorm layer and ReLU activation
    """
    def ConvUnit(inputs, layer_index):
        x = Conv2D(filters, 3, padding='same', name='conv_'+layer_index, use_bias=not(batchnorm), kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(inputs)
        if batchnorm:
            x = BatchNormalization(name='bn_'+layer_index)(x)
        return Activation('relu', name='relu_'+layer_index)(x)

    x = ConvUnit(inputs, index+'a')
    return ConvUnit(x, index+'b')


def UpsampleBlock(inputs, filters, batchnorm, l2_reg, index):
    """
    A Conv2DTranspose layer for upsampling with optional batchnorm layer and ReLU activation
    """
    x = Conv2DTranspose(filters, 2, padding='same', name='up_'+index, use_bias=not(batchnorm), kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), strides=2)(inputs)
    if batchnorm:
        x = BatchNormalization(name='bn_'+index)(x)
    return Activation('relu', name='relu_'+index)(x)