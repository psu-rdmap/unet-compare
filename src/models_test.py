from tensorflow.keras.layers import *
from keras.regularizers import l2
from keras.models import Model






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


def unet(IMG_SIZE, l2_reg, batchnorm=True):
    filters = [32, 64, 128, 256, 512]

    # input
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='main_input')

    # -- Encoder -- #
    conv1_1 = encoder_unit(inputs, '11', filters[0], l2_reg, batchnorm=batchnorm)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1_1)

    conv2_1 = encoder_unit(pool1, '21', filters[1], l2_reg, batchnorm=batchnorm)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2_1)

    conv3_1 = encoder_unit(pool2, '31', filters[2], l2_reg, batchnorm=batchnorm)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3_1)

    conv4_1 = encoder_unit(pool3, '41', filters[3], l2_reg, batchnorm=batchnorm)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4_1)

    conv5_1 = encoder_unit(pool4, '51', filters[4], l2_reg, batchnorm=batchnorm)

    # -- Decoder -- #
    up4_2 = decoder_unit(conv5_1, '42', filters[3], l2_reg, kernel_size=2, batchnorm=batchnorm)
    conv4_2 = concatenate([conv4_1, up4_2], name='merge42', axis=3)
    conv4_2 = encoder_unit(conv4_2, '42', filters[3], l2_reg, batchnorm=batchnorm)

    up3_3 = decoder_unit(conv4_2, '33', filters[2], l2_reg, kernel_size=2, batchnorm=batchnorm)
    conv3_3 = concatenate([conv3_1, up3_3], name='merge33', axis=3)
    conv3_3 = encoder_unit(conv3_3, '33', filters[2], l2_reg, batchnorm=batchnorm)

    up2_4 = decoder_unit(conv3_3, '24', filters[1], l2_reg, kernel_size=2, batchnorm=batchnorm)
    conv2_4 = concatenate([conv2_1, up2_4], name='merge24', axis=3)
    conv2_4 = encoder_unit(conv2_4, '24', filters[1], l2_reg, batchnorm=batchnorm)

    up1_5 = decoder_unit(conv2_4, '15', filters[0], l2_reg, kernel_size=2, batchnorm=batchnorm)
    conv1_5 = concatenate([conv1_1, up1_5], name='merge15', axis=3)
    conv1_5 = encoder_unit(conv1_5, '15', filters[0], l2_reg, batchnorm=batchnorm)

    # output
    output = Conv2D(1, 1, activation='sigmoid', name='output', kernel_initializer='he_normal', padding='same',
                    kernel_regularizer=l2(l2_reg))(conv1_5)
    model = Model(inputs=inputs, outputs=output)

    return model
