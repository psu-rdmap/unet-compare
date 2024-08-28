import keras
import tensorflow as tf
from tensorflow.keras.layers import *
from keras.regularizers import l2
from keras.models import Model
from tensorflow.keras.applications import EfficientNetB7, VGG16
from blocks import encoder_unit, decoder_unit
  
def unet(IMG_SIZE, l2_reg, batchnorm=True):
    # filters numbers
    filters = [32, 64, 128, 256, 512]
    #filters = [16, 32, 64, 128, 512]
    #filters = [8, 16, 32, 64, 128]
    print('\nU-Net Filters:', filters)

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


def unet_eff(IMG_SIZE, l2_reg, batchnorm=True):
    N_CLASSES = 1
    
    # color channels must be 3 for EfficientNet encoder
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    input_tensor = Input(shape=input_shape, name='main_input')

    # encoder instantiation
    backbone = EfficientNetB7(include_top=False,
                              weights='imagenet',
                              input_shape=input_shape,
                              input_tensor=input_tensor,
                              classes=N_CLASSES)

    # encoder output stages
    encoder_outputs = ['stem_activation', 'block2g_add', 'block3g_add', 'block5j_add', 'block7d_add']

    # get feature maps from various encoder stages
    outputs_dict = {layer: backbone.get_layer(layer).output for layer in encoder_outputs}
    encoder = Model(inputs=backbone.input, outputs=outputs_dict)
    enc_maps = encoder(input_tensor)

    # number of filters from encoder stages and for decoding
    enc_filters = [64, 48, 80, 224, 640]
    dec_filters = [256, 128, 64, 32, 16]

    # -- Encoder -- #
    
    enc_0_out = enc_maps[encoder_outputs[0]]
    
    enc_1_out = enc_maps[encoder_outputs[1]]
    
    enc_2_out = enc_maps[encoder_outputs[2]]
   
    enc_3_out = enc_maps[encoder_outputs[3]]
    
    enc_4_out = enc_maps[encoder_outputs[4]]

    # -- Decoder -- #

    up3_1 = decoder_unit(enc_4_out, '31', enc_filters[4], l2_reg, kernel_size=2, batchnorm=batchnorm)

    conv3_1 = concatenate([up3_1, enc_3_out], name='merge31', axis=3)
    conv3_1 = encoder_unit(conv3_1, '31', dec_filters[0], l2_reg, batchnorm=batchnorm)
    up2_2 = decoder_unit(conv3_1, '22', dec_filters[0], l2_reg, kernel_size=2, batchnorm=batchnorm)
    
    conv2_2 = concatenate([up2_2, enc_2_out], name='merge22', axis=3)
    conv2_2 = encoder_unit(conv2_2, '22', dec_filters[1], l2_reg, batchnorm=batchnorm)
    up1_3 = decoder_unit(conv2_2, '13', dec_filters[1], l2_reg, kernel_size=2, batchnorm=batchnorm)
    
    
    conv1_3 = concatenate([up1_3, enc_1_out], name='merge13', axis=3)
    conv1_3 = encoder_unit(conv1_3, '13', dec_filters[2], l2_reg, batchnorm=batchnorm)
    up0_4 = decoder_unit(conv1_3, '04', dec_filters[2], l2_reg, kernel_size=2, batchnorm=batchnorm)

    conv0_4 = concatenate([up0_4, enc_0_out], name='merge04', axis=3)
    conv0_4 = encoder_unit(conv0_4, '04', dec_filters[3], l2_reg, batchnorm=batchnorm)

    up_out = decoder_unit(conv0_4, '_out', dec_filters[4], l2_reg, batchnorm=batchnorm)
        
    output = Conv2D(1, 1, activation='sigmoid', name='output', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(l2_reg))(up_out)
    
    model = Model(inputs=input_tensor, outputs=output)

    return model


def unet_vgg16(IMG_SIZE, l2_reg, batchnorm=True):
    N_CLASSES = 1
    
    # color channels must be 3 for EfficientNet encoder
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    input_tensor = Input(shape=input_shape, name='main_input')

    # encoder instantiation
    backbone = VGG16(include_top=False,
                     weights='imagenet',
                     input_shape=input_shape,
                     input_tensor=input_tensor,
                     classes=N_CLASSES)

    # encoder output stages
    encoder_outputs = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool']

    # get feature maps from various encoder stages
    outputs_dict = {layer: backbone.get_layer(layer).output for layer in encoder_outputs}
    encoder = Model(inputs=backbone.input, outputs=outputs_dict)
    enc_maps = encoder(input_tensor)

    # number of filters from encoder stages and for decoding
    enc_filters = [64, 128, 256, 512, 512]
    dec_filters = [256, 128, 64, 32, 16]

    # -- Encoder -- #
    
    enc_0_out = enc_maps[encoder_outputs[0]]
    
    enc_1_out = enc_maps[encoder_outputs[1]]
    
    enc_2_out = enc_maps[encoder_outputs[2]]
   
    enc_3_out = enc_maps[encoder_outputs[3]]
    
    enc_4_out = enc_maps[encoder_outputs[4]]

    # -- Decoder -- #

    up3_1 = decoder_unit(enc_4_out, '31', enc_filters[4], l2_reg, kernel_size=2, batchnorm=batchnorm)

    conv3_1 = concatenate([up3_1, enc_3_out], name='merge31', axis=3)
    conv3_1 = encoder_unit(conv3_1, '31', dec_filters[0], l2_reg, batchnorm=batchnorm)
    up2_2 = decoder_unit(conv3_1, '22', dec_filters[0], l2_reg, kernel_size=2, batchnorm=batchnorm)
    
    conv2_2 = concatenate([up2_2, enc_2_out], name='merge22', axis=3)
    conv2_2 = encoder_unit(conv2_2, '22', dec_filters[1], l2_reg, batchnorm=batchnorm)
    up1_3 = decoder_unit(conv2_2, '13', dec_filters[1], l2_reg, kernel_size=2, batchnorm=batchnorm)
    
    
    conv1_3 = concatenate([up1_3, enc_1_out], name='merge13', axis=3)
    conv1_3 = encoder_unit(conv1_3, '13', dec_filters[2], l2_reg, batchnorm=batchnorm)
    up0_4 = decoder_unit(conv1_3, '04', dec_filters[2], l2_reg, kernel_size=2, batchnorm=batchnorm)

    conv0_4 = concatenate([up0_4, enc_0_out], name='merge04', axis=3)
    conv0_4 = encoder_unit(conv0_4, '04', dec_filters[3], l2_reg, batchnorm=batchnorm)

    up_out = decoder_unit(conv0_4, '_out', dec_filters[4], l2_reg, batchnorm=batchnorm)
        
    output = Conv2D(1, 1, activation='sigmoid', name='output', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(l2_reg))(up_out)
    
    model = Model(inputs=input_tensor, outputs=output)

    return model



def unetpp(IMG_SIZE, l2_reg, batchnorm=True, deep_supervision = False):
    # filters numbers
    filters = [32, 64, 128, 256, 512]

    # input
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='main_input')

    # -- U-Net 1 -- #

    # block (0,0)
    conv0_0 = encoder_unit(inputs, '00', filters[0], l2_reg, batchnorm=batchnorm)
    # down (0,0)->(1,0)
    pool0 = MaxPooling2D(pool_size=(2, 2), name='pool0')(conv0_0)
    # block (1,0)
    conv1_0 = encoder_unit(pool0, '10', filters[1], l2_reg, batchnorm=batchnorm)
    # up (1,0)->(0,1)
    up0_1 = decoder_unit(conv1_0, '01', filters[0], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (0,0),(1,0)->(0,1)
    conv0_1 = concatenate([up0_1, conv0_0], name='merge01', axis=3)
    # block (0,1)
    conv0_1 = encoder_unit(conv0_1, '01', filters[0], l2_reg, batchnorm=batchnorm)

    # -- U-Net 2 -- #

    # down (1,0)->(2,0)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1_0)
    # block (2,0)
    conv2_0 = encoder_unit(pool1, '20', filters[2], l2_reg, batchnorm=batchnorm)
    # up (2,0)->(1,1)
    up1_1 = decoder_unit(conv2_0, '11', filters[1], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (1,0),(2,0)->(1,1)
    conv1_1 = concatenate([up1_1, conv1_0], name='merge11', axis=3)
    # block (1,1)
    conv1_1 = encoder_unit(conv1_1, '11', filters[1], l2_reg, batchnorm=batchnorm)
    # up (1,1)->(0,2)
    up0_2 = decoder_unit(conv1_1, '02', filters[0], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (0,0),(0,1),(1,1)->(0,2)
    conv0_2 = concatenate([up0_2, conv0_0, conv0_1], name='merge02', axis=3)
    # block (0,2)
    conv0_2 = encoder_unit(conv0_2, '02', filters[0], l2_reg, batchnorm=batchnorm)

    # -- U-Net 3 -- #

    # down (2,0)->(3,0)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2_0)
    # block (3,0)
    conv3_0 = encoder_unit(pool2, '30', filters[3], l2_reg, batchnorm=batchnorm)
    # up (3,0)->(2,1)
    up2_1 = decoder_unit(conv3_0, '21', filters[2], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (2,0),(3,0)->(2,1)
    conv2_1 = concatenate([up2_1, conv2_0], name='merge21', axis=3)
    # block (2,1)
    conv2_1 = encoder_unit(conv2_1, '21', filters[2], l2_reg, batchnorm=batchnorm)
    # up (2,1)->(1,2)
    up1_2 = decoder_unit(conv2_1, '12', filters[1], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (1,0),(1,1),(2,1)->(1,2)
    conv1_2 = concatenate([up1_2, conv1_0, conv1_1], name='merge12', axis=3)
    # block (1,2)
    conv1_2 = encoder_unit(conv1_2, '12', filters[1], l2_reg, batchnorm=batchnorm)
    # up (1,2)->(0,3)
    up0_3 = decoder_unit(conv1_2, '03', filters[0], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (0,0),(0,1),(0,2),(1,2)->(0,3)
    conv0_3 = concatenate([up0_3, conv0_0, conv0_1, conv0_2], name='merge03', axis=3)
    # block (0,3)
    conv0_3 = encoder_unit(conv0_3, '03', filters[0], l2_reg, batchnorm=batchnorm)

    # -- U-Net 4 -- #

    # down (3,0)->(4,0)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3_0)
    # block (4,0)
    conv4_0 = encoder_unit(pool3, '40', filters[4], l2_reg, batchnorm=batchnorm)
    # up (4,0)->(3,1)
    up3_1 = decoder_unit(conv4_0, '31', filters[3], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (3,0),(4,0)->(3,1)
    conv3_1 = concatenate([up3_1, conv3_0], name='merge31', axis=3)
    # block (3,1)
    conv3_1 = encoder_unit(conv3_1, '31', filters[3], l2_reg, batchnorm=batchnorm)
    # up (3,1)->(2,2)
    up2_2 = decoder_unit(conv3_1, '22', filters[2], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (2,0),(2,1),(3,1)->(2,2)
    conv2_2 = concatenate([up2_2, conv2_0, conv2_1], name='merge22', axis=3)
    # block (2,2)
    conv2_2 = encoder_unit(conv2_2, '22', filters[2], l2_reg, batchnorm=batchnorm)
    # up (2,2)->(1,3)
    up1_3 = decoder_unit(conv2_2, '13', filters[1], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (1,0),(1,1),(1,2),(2,2)->(1,3)
    conv1_3 = concatenate([up1_3, conv1_0, conv1_1, conv1_2], name='merge13', axis=3)
    # block (1,3)
    conv1_3 = encoder_unit(conv1_3, '13', filters[1], l2_reg, batchnorm=batchnorm)
    # up (1,3)->(0,4)
    up0_4 = decoder_unit(conv1_3, '04', filters[0], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (0,0),(0,1),(0,2),(0,3),(1,3)->(0,4)
    conv0_4 = concatenate([up0_4, conv0_0, conv0_1, conv0_2, conv0_3], name='merge04', axis=3)
    # block (0,4)
    conv0_4 = encoder_unit(conv0_4, '04', filters[0], l2_reg, batchnorm=batchnorm)

    # nested unet outputs
    nestnet_output_1 = Conv2D(1, 1, activation='sigmoid', name='output_1', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(l2_reg))(conv0_1)
    nestnet_output_2 = Conv2D(1, 1, activation='sigmoid', name='output_2', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(l2_reg))(conv0_2)
    nestnet_output_3 = Conv2D(1, 1, activation='sigmoid', name='output_3', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(l2_reg))(conv0_3)
    nestnet_output_4 = Conv2D(1, 1, activation='sigmoid', name='output_4', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(l2_reg))(conv0_4)

    if deep_supervision:
        model = Model(inputs=inputs, outputs=[nestnet_output_1, nestnet_output_2, nestnet_output_3, nestnet_output_4])
    else:
        model = Model(inputs=inputs, outputs=nestnet_output_4)

    return model


def unetpp_eff(IMG_SIZE, l2_reg, batchnorm=True, deep_supervision=False):
    N_CLASSES = 1
    
    # color channels must be 3 for EfficientNet encoder
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    input_tensor = Input(shape=input_shape, name='main_input')

    # encoder instantiation
    backbone = EfficientNetB7(include_top=False,
                              weights='imagenet',
                              input_shape=input_shape,
                              input_tensor=input_tensor,
                              classes=N_CLASSES)

    # encoder output stages
    encoder_outputs = ['stem_activation', 'block2g_add', 'block3g_add', 'block5j_add', 'block7d_add']

    # get feature maps from various encoder stages
    outputs_dict = {layer: backbone.get_layer(layer).output for layer in encoder_outputs}
    encoder = Model(inputs=backbone.input, outputs=outputs_dict)
    enc_maps = encoder(input_tensor)

    # number of filterss from encoder stages and for decoding
    enc_filters = [64, 48, 80, 224, 640]
    dec_filters = [256, 128, 64, 32, 16]

    # -- U-Net 1 -- #

    # enc stage 0 output (0,0)
    enc_0_out = enc_maps[encoder_outputs[0]]
    # enc stage 1 output (1,0)
    enc_1_out = enc_maps[encoder_outputs[1]]
    # up (1,0)->(0,1)
    up0_1 = decoder_unit(enc_1_out, '01', enc_filters[0], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (0,0),(1,0)->(0,1)
    conv0_1 = concatenate([up0_1, enc_0_out], name='merge01', axis=3)
    # block (0,1)
    conv0_1 = encoder_unit(conv0_1, '01', enc_filters[0], l2_reg, batchnorm=batchnorm)

    # -- U-Net 2 -- #

    # enc stage 2 output (2,0)
    enc_2_out = enc_maps[encoder_outputs[2]]
    # up (2,0)->(1,1)
    up1_1 = decoder_unit(enc_2_out, '11', enc_filters[1], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (1,0),(2,0)->(1,1)
    conv1_1 = concatenate([up1_1, enc_1_out], name='merge11', axis=3)
    # block (1,1)
    conv1_1 = encoder_unit(conv1_1, '11', enc_filters[1], l2_reg, batchnorm=batchnorm)
    # up (1,1)->(0,2)
    up0_2 = decoder_unit(conv1_1, '02', enc_filters[0], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (0,0),(0,1),(1,1)->(0,2)
    conv0_2 = concatenate([up0_2, enc_0_out, conv0_1], name='merge02', axis=3)
    # block (0,2)
    conv0_2 = encoder_unit(conv0_2, '02', enc_filters[0], l2_reg, batchnorm=batchnorm)

    # -- U-Net 3 -- #

    # enc stage 3 output (3,0)
    enc_3_out = enc_maps[encoder_outputs[3]]
    # up (3,0)->(2,1)
    up2_1 = decoder_unit(enc_3_out, '21', enc_filters[2], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (2,0),(3,0)->(2,1)
    conv2_1 = concatenate([up2_1, enc_2_out], name='merge21', axis=3)
    # block (2,1)
    conv2_1 = encoder_unit(conv2_1, '21', enc_filters[2], l2_reg, batchnorm=batchnorm)
    # up (2,1)->(1,2)
    up1_2 = decoder_unit(conv2_1, '12', enc_filters[1], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (1,0),(1,1),(2,1)->(1,2)
    conv1_2 = concatenate([up1_2, enc_1_out, conv1_1], name='merge12', axis=3)
    # block (1,2)
    conv1_2 = encoder_unit(conv1_2, '12', enc_filters[1], l2_reg, batchnorm=batchnorm)
    # up (1,2)->(0,3)
    up0_3 = decoder_unit(conv1_2, '03', enc_filters[0], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (0,0),(0,1),(0,2),(1,2)->(0,3)
    conv0_3 = concatenate([up0_3, enc_0_out, conv0_1, conv0_2], name='merge03', axis=3)
    # block (0,3)
    conv0_3 = encoder_unit(conv0_3, '03', enc_filters[0], l2_reg, batchnorm=batchnorm)

    # -- U-Net 4 -- #

    # enc stage 4 output (4,0)
    enc_4_out = enc_maps[encoder_outputs[4]]
    # up (4,0)->(3,1)
    up3_1 = decoder_unit(enc_4_out, '31', enc_filters[4], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (3,0),(4,0)->(3,1)
    conv3_1 = concatenate([up3_1, enc_3_out], name='merge31', axis=3)
    # block (3,1)
    conv3_1 = encoder_unit(conv3_1, '31', dec_filters[0], l2_reg, batchnorm=batchnorm)
    # up (3,1)->(2,2)
    up2_2 = decoder_unit(conv3_1, '22', dec_filters[0], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (2,0),(2,1),(3,1)->(2,2)
    conv2_2 = concatenate([up2_2, enc_2_out, conv2_1], name='merge22', axis=3)
    # block (2,2)
    conv2_2 = encoder_unit(conv2_2, '22', dec_filters[1], l2_reg, batchnorm=batchnorm)
    # up (2,2)->(1,3)
    up1_3 = decoder_unit(conv2_2, '13', dec_filters[1], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (1,0),(1,1),(1,2),(2,2)->(1,3)
    conv1_3 = concatenate([up1_3, enc_1_out, conv1_1, conv1_2], name='merge13', axis=3)
    # block (1,3)
    conv1_3 = encoder_unit(conv1_3, '13', dec_filters[2], l2_reg, batchnorm=batchnorm)
    # up (1,3)->(0,4)
    up0_4 = decoder_unit(conv1_3, '04', dec_filters[2], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (0,0),(0,1),(0,2),(0,3),(1,3)->(0,4)
    conv0_4 = concatenate([up0_4, enc_0_out, conv0_1, conv0_2, conv0_3], name='merge04', axis=3)
    # block (0,4)
    conv0_4 = encoder_unit(conv0_4, '04', dec_filters[3], l2_reg, batchnorm=batchnorm)

    up_1 = decoder_unit(conv0_1, '_out_1', dec_filters[4], l2_reg, batchnorm=batchnorm)
    up_2 = decoder_unit(conv0_2, '_out_2', dec_filters[4], l2_reg, batchnorm=batchnorm)
    up_3 = decoder_unit(conv0_3, '_out_3', dec_filters[4], l2_reg, batchnorm=batchnorm)
    up_4 = decoder_unit(conv0_4, '_out_4', dec_filters[4], l2_reg, batchnorm=batchnorm)

    nestnet_output_1 = Conv2D(1, 1, activation='sigmoid', name='output_1', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(l2_reg))(up_1)
    nestnet_output_2 = Conv2D(1, 1, activation='sigmoid', name='output_2', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(l2_reg))(up_2)
    nestnet_output_3 = Conv2D(1, 1, activation='sigmoid', name='output_3', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(l2_reg))(up_3)
    nestnet_output_4 = Conv2D(1, 1, activation='sigmoid', name='output_4', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(l2_reg))(up_4)

    if deep_supervision:
        model = Model(inputs=input_tensor,
                      outputs=[nestnet_output_1, nestnet_output_2, nestnet_output_3, nestnet_output_4])
    else:
        model = Model(inputs=input_tensor, outputs=nestnet_output_4)

    return model

 
def unetpp_vgg16(IMG_SIZE, l2_reg, batchnorm=True, deep_supervision=False):
    N_CLASSES = 1
    
    # color channels must be 3 for EfficientNet encoder
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    input_tensor = Input(shape=input_shape, name='main_input')

    # encoder instantiation
    backbone = VGG16(include_top=False,
                              weights='imagenet',
                              input_shape=input_shape,
                              input_tensor=input_tensor,
                              classes=N_CLASSES)

    # encoder output stages
    encoder_outputs = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool']

    # get feature maps from various encoder stages
    outputs_dict = {layer: backbone.get_layer(layer).output for layer in encoder_outputs}
    encoder = Model(inputs=backbone.input, outputs=outputs_dict)
    enc_maps = encoder(input_tensor)

    # number of filterss from encoder stages and for decoding
    enc_filters = [64, 48, 80, 224, 640]
    dec_filters = [256, 128, 64, 32, 16]

    # -- U-Net 1 -- #

    # enc stage 0 output (0,0)
    enc_0_out = enc_maps[encoder_outputs[0]]
    # enc stage 1 output (1,0)
    enc_1_out = enc_maps[encoder_outputs[1]]
    # up (1,0)->(0,1)
    up0_1 = decoder_unit(enc_1_out, '01', enc_filters[0], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (0,0),(1,0)->(0,1)
    conv0_1 = concatenate([up0_1, enc_0_out], name='merge01', axis=3)
    # block (0,1)
    conv0_1 = encoder_unit(conv0_1, '01', enc_filters[0], l2_reg, batchnorm=batchnorm)

    # -- U-Net 2 -- #

    # enc stage 2 output (2,0)
    enc_2_out = enc_maps[encoder_outputs[2]]
    # up (2,0)->(1,1)
    up1_1 = decoder_unit(enc_2_out, '11', enc_filters[1], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (1,0),(2,0)->(1,1)
    conv1_1 = concatenate([up1_1, enc_1_out], name='merge11', axis=3)
    # block (1,1)
    conv1_1 = encoder_unit(conv1_1, '11', enc_filters[1], l2_reg, batchnorm=batchnorm)
    # up (1,1)->(0,2)
    up0_2 = decoder_unit(conv1_1, '02', enc_filters[0], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (0,0),(0,1),(1,1)->(0,2)
    conv0_2 = concatenate([up0_2, enc_0_out, conv0_1], name='merge02', axis=3)
    # block (0,2)
    conv0_2 = encoder_unit(conv0_2, '02', enc_filters[0], l2_reg, batchnorm=batchnorm)

    # -- U-Net 3 -- #

    # enc stage 3 output (3,0)
    enc_3_out = enc_maps[encoder_outputs[3]]
    # up (3,0)->(2,1)
    up2_1 = decoder_unit(enc_3_out, '21', enc_filters[2], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (2,0),(3,0)->(2,1)
    conv2_1 = concatenate([up2_1, enc_2_out], name='merge21', axis=3)
    # block (2,1)
    conv2_1 = encoder_unit(conv2_1, '21', enc_filters[2], l2_reg, batchnorm=batchnorm)
    # up (2,1)->(1,2)
    up1_2 = decoder_unit(conv2_1, '12', enc_filters[1], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (1,0),(1,1),(2,1)->(1,2)
    conv1_2 = concatenate([up1_2, enc_1_out, conv1_1], name='merge12', axis=3)
    # block (1,2)
    conv1_2 = encoder_unit(conv1_2, '12', enc_filters[1], l2_reg, batchnorm=batchnorm)
    # up (1,2)->(0,3)
    up0_3 = decoder_unit(conv1_2, '03', enc_filters[0], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (0,0),(0,1),(0,2),(1,2)->(0,3)
    conv0_3 = concatenate([up0_3, enc_0_out, conv0_1, conv0_2], name='merge03', axis=3)
    # block (0,3)
    conv0_3 = encoder_unit(conv0_3, '03', enc_filters[0], l2_reg, batchnorm=batchnorm)

    # -- U-Net 4 -- #

    # enc stage 4 output (4,0)
    enc_4_out = enc_maps[encoder_outputs[4]]
    # up (4,0)->(3,1)
    up3_1 = decoder_unit(enc_4_out, '31', enc_filters[4], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (3,0),(4,0)->(3,1)
    conv3_1 = concatenate([up3_1, enc_3_out], name='merge31', axis=3)
    # block (3,1)
    conv3_1 = encoder_unit(conv3_1, '31', dec_filters[0], l2_reg, batchnorm=batchnorm)
    # up (3,1)->(2,2)
    up2_2 = decoder_unit(conv3_1, '22', dec_filters[0], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (2,0),(2,1),(3,1)->(2,2)
    conv2_2 = concatenate([up2_2, enc_2_out, conv2_1], name='merge22', axis=3)
    # block (2,2)
    conv2_2 = encoder_unit(conv2_2, '22', dec_filters[1], l2_reg, batchnorm=batchnorm)
    # up (2,2)->(1,3)
    up1_3 = decoder_unit(conv2_2, '13', dec_filters[1], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (1,0),(1,1),(1,2),(2,2)->(1,3)
    conv1_3 = concatenate([up1_3, enc_1_out, conv1_1, conv1_2], name='merge13', axis=3)
    # block (1,3)
    conv1_3 = encoder_unit(conv1_3, '13', dec_filters[2], l2_reg, batchnorm=batchnorm)
    # up (1,3)->(0,4)
    up0_4 = decoder_unit(conv1_3, '04', dec_filters[2], l2_reg, kernel_size=2, batchnorm=batchnorm)
    # merge (0,0),(0,1),(0,2),(0,3),(1,3)->(0,4)
    conv0_4 = concatenate([up0_4, enc_0_out, conv0_1, conv0_2, conv0_3], name='merge04', axis=3)
    # block (0,4)
    conv0_4 = encoder_unit(conv0_4, '04', dec_filters[3], l2_reg, batchnorm=batchnorm)

    up_1 = decoder_unit(conv0_1, '_out_1', dec_filters[4], l2_reg, batchnorm=batchnorm)
    up_2 = decoder_unit(conv0_2, '_out_2', dec_filters[4], l2_reg, batchnorm=batchnorm)
    up_3 = decoder_unit(conv0_3, '_out_3', dec_filters[4], l2_reg, batchnorm=batchnorm)
    up_4 = decoder_unit(conv0_4, '_out_4', dec_filters[4], l2_reg, batchnorm=batchnorm)

    nestnet_output_1 = Conv2D(1, 1, activation='sigmoid', name='output_1', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(l2_reg))(up_1)
    nestnet_output_2 = Conv2D(1, 1, activation='sigmoid', name='output_2', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(l2_reg))(up_2)
    nestnet_output_3 = Conv2D(1, 1, activation='sigmoid', name='output_3', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(l2_reg))(up_3)
    nestnet_output_4 = Conv2D(1, 1, activation='sigmoid', name='output_4', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(l2_reg))(up_4)

    if deep_supervision:
        model = Model(inputs=input_tensor,
                      outputs=[nestnet_output_1, nestnet_output_2, nestnet_output_3, nestnet_output_4])
    else:
        model = Model(inputs=input_tensor, outputs=nestnet_output_4)

    return model

 
