import keras
from keras import layers
import tensorflow as tf
from tensorflow.keras.layers import *
from keras.regularizers import l2

# single Conv2D layer with
#     batchnorm (and no bias if used)
#     L2 regularization
#     ReLU activation
#     He normal initialization
#     name indices
def ConvRelu(input, filter, kernel_size, l2_reg, name_index, use_batchnorm=False):
    x = Conv2D(filter, kernel_size, padding="same", name='conv'+name_index, use_bias=not(use_batchnorm), kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(input)
    if use_batchnorm:
        x = BatchNormalization(name='bn'+name_index)(x)
    x = Activation('relu', name='relu'+name_index)(x)
    return x

# single Conv2DTranspose layer with
#     batchnorm (and no bias if used)
#     L2 regularization
#     ReLU activation
#     He normal initialization
#     name
def ConvReluDecoder(input, filter, kernel_size, l2_reg, name_index, use_batchnorm=False):
    x = Conv2DTranspose(filter, kernel_size, strides=(2,2), padding="same", name='up'+name_index, use_bias=not(use_batchnorm), kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(input)
    if use_batchnorm:
        x = BatchNormalization(name='bn'+name_index)(x)
    x = Activation('relu', name='relu'+name_index)(x)
    return x

# define standard convolution block (2 conv blocks)
#     input tensor
#     L2 regularization parameter
#     stage in model for naming
#     filters
#     kernel size
#     option to use batchnorm
def encoder_unit(input, stage, filter, l2_reg, kernel_size=3, batchnorm=False):
    x = ConvRelu(input, filter, kernel_size, l2_reg, name_index=stage+'_1', use_batchnorm=batchnorm)
    x = ConvRelu(x,            filter, kernel_size, l2_reg, name_index=stage+'_2', use_batchnorm=batchnorm)
    return x

# define standard up convolution (decoder) block
def decoder_unit(input, name_index, filter, l2_reg, kernel_size=2, batchnorm=False, decoder_type=1):
    if decoder_type: # Conv2DTranspose
        x = ConvReluDecoder(input, filter, kernel_size, l2_reg, name_index, use_batchnorm=batchnorm)

    else: # UpSample -> Conv2D
        x = Conv2D(filter, kernel_size, name='up'+name_index, padding='same', kernel_initializer='he_normal', use_bias=not(batchnorm), kernel_regularizer=l2(l2_reg))(UpSampling2D(size=(2, 2), name='upsample'+name_index)(input))
        if batchnorm:
            x = BatchNormalization(name='bn_up'+name_index)(x)
        x = Activation('relu', name='relu_up'+name_index)(x)

    return x