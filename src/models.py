"""
This module handles the definition of encoder/decoder subnetworks and their connection
"""

import keras
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Concatenate
from keras.regularizers import l2
from blocks import ConvBlock, UpsampleUnit
from tensorflow.keras.applications import EfficientNetB7


class Encoder(tf.keras.Model):
    def __init__(self, configs : dict):
        super().__init__()
        
        # unpack configs and assign them as instance attributes
        for key, value in configs.items():
            setattr(self, key, value)

        self.build()

    def build(self):
        encoder_input = keras.Input(self.input_shape, name='main_input')
        
        if self.encoder_name == 'Vanilla':
            conv00 = ConvBlock(self.encoder_filters[0], self.batchnorm, self.l2_reg, name = 'conv_block_00')(encoder_input)
            pool0 = MaxPooling2D(pool_size=2, name='pool_0')(conv00)

            conv10 = ConvBlock(self.encoder_filters[1], self.batchnorm, self.l2_reg, name = 'conv_block_10')(pool0)
            pool1 = MaxPooling2D(pool_size=2, name='pool_1')(conv10)

            conv20 = ConvBlock(self.encoder_filters[2], self.batchnorm, self.l2_reg, name = 'conv_block_20')(pool1)
            pool2 = MaxPooling2D(pool_size=2, name='pool_2')(conv20)

            conv30 = ConvBlock(self.encoder_filters[3], self.batchnorm, self.l2_reg, name = 'conv_block_30')(pool2)
            pool3 = MaxPooling2D(pool_size=2, name='pool_3')(conv30)

            conv40 = ConvBlock(self.encoder_filters[4], self.batchnorm, self.l2_reg, name = 'conv_block_40')(pool3)
            
            self.encoder = keras.Model(inputs=encoder_input, outputs=conv40, name = 'UNet Vanilla Encoder')

        elif self.encoder_name == 'EfficientNetB7':
            self.encoder = EfficientNetB7(
                include_top  = False,
                weights      = 'imagenet',
                input_tensor  = encoder_input,
        )
    
        else:
            raise ValueError('{} is not a valid encoder name'.format(self.encoder_name))

    def call(self):
        return self.encoder()
    

class Decoder(tf.keras.Model):
    def __init__(self, configs : dict):
        super().__init__()
        
        # unpack configs and assign them as instance attributes
        for key, value in configs.items():
            setattr(self, key, value)

        # instantiate encoder
        self.encoder = Encoder(configs).encoder

        self.build()

    def build(self):
        encoder_outputs = [self.encoder.get_layer(stage).output for stage in self.encoder_stages]
    
        if self.decoder_name == 'UNet':
            up3_1 = UpsampleUnit(self.decoder_filters[0], self.batchnorm, self.l2_reg, name='upsample_31')(encoder_outputs[4])
            cat3_1 = Concatenate(name='cat_31')([up3_1, encoder_outputs[3]])
            conv3_1 = ConvBlock(self.decoder_filters[0], batchnorm=self.batchnorm, l2_reg=self.l2_reg, name='conv_block_31')(cat3_1)
            
            up2_2 = UpsampleUnit(self.decoder_filters[1], self.batchnorm, self.l2_reg, name='upsample_22')(conv3_1)
            cat2_2 = Concatenate(name='cat_22')([up2_2, encoder_outputs[2]])
            conv2_2 = ConvBlock(self.decoder_filters[1], batchnorm=self.batchnorm, l2_reg=self.l2_reg, name='conv_block_22')(cat2_2)

            up1_3 = UpsampleUnit(self.decoder_filters[2], self.batchnorm, self.l2_reg, name='upsample_13')(conv2_2)
            cat1_3 = Concatenate(name='cat_13')([up1_3, encoder_outputs[1]])
            conv1_3 = ConvBlock(self.decoder_filters[2], batchnorm=self.batchnorm, l2_reg=self.l2_reg, name='conv_block_13')(cat1_3)

            up0_4 = UpsampleUnit(self.decoder_filters[3], self.batchnorm, self.l2_reg, name='upsample_04')(conv1_3)
            cat0_4 = Concatenate(name='cat_04')([up0_4, encoder_outputs[0]])
            conv0_4 = ConvBlock(self.decoder_filters[3], batchnorm=self.batchnorm, l2_reg=self.l2_reg, name='conv_block_04')(cat0_4)
                
            if self.encoder_name == 'EfficientNetB7':
                up_out = UpsampleUnit(self.decoder_filters[4], self.batchnorm, self.l2_reg, name='upsample_out')(conv0_4)
                last_unit = up_out
            else:
                last_unit = conv0_4

            output = Conv2D(1, 1, activation='sigmoid', name='output', kernel_initializer='he_normal',
                                        padding='same', kernel_regularizer=l2(self.l2_reg))(last_unit)

            self.decoder = keras.Model(inputs=self.encoder.input, outputs=output, name='UNet')


        elif self.decoder_name == 'UNet++':
            # column 1
            up0_1 = UpsampleUnit(self.decoder_filters[3], self.batchnorm, self.l2_reg, name='upsample_01')(encoder_outputs[1])
            cat0_1 = Concatenate(name='cat_01')([up0_1, encoder_outputs[0]])
            conv0_1 = ConvBlock(self.decoder_filters[3], batchnorm=self.batchnorm, l2_reg=self.l2_reg, name='conv_block_01')(cat0_1)

            up1_1 = UpsampleUnit(self.decoder_filters[2], self.batchnorm, self.l2_reg, name='upsample_11')(encoder_outputs[2])
            cat1_1 = Concatenate(name='cat_11')([up1_1, encoder_outputs[1]])
            conv1_1 = ConvBlock(self.decoder_filters[2], batchnorm=self.batchnorm, l2_reg=self.l2_reg, name='conv_block_11')(cat1_1)

            up2_1 = UpsampleUnit(self.decoder_filters[1], self.batchnorm, self.l2_reg, name='upsample_21')(encoder_outputs[3])
            cat2_1 = Concatenate(name='cat_21')([up2_1, encoder_outputs[2]])
            conv2_1 = ConvBlock(self.decoder_filters[1], batchnorm=self.batchnorm, l2_reg=self.l2_reg, name='conv_block_21')(cat2_1)

            up3_1 = UpsampleUnit(self.decoder_filters[0], self.batchnorm, self.l2_reg, name='upsample_31')(encoder_outputs[4])
            cat3_1 = Concatenate(name='cat_31')([up3_1, encoder_outputs[3]])
            conv3_1 = ConvBlock(self.decoder_filters[0], batchnorm=self.batchnorm, l2_reg=self.l2_reg, name='conv_block_31')(cat3_1)

            # column 2
            up0_2 = UpsampleUnit(self.decoder_filters[3], self.batchnorm, self.l2_reg, name='upsample_02')(conv1_1)
            cat0_2 = Concatenate(name='cat_02')([up0_2, conv0_1, encoder_outputs[0]])
            conv0_2 = ConvBlock(self.decoder_filters[3], batchnorm=self.batchnorm, l2_reg=self.l2_reg, name='conv_block_02')(cat0_2)

            up1_2 = UpsampleUnit(self.decoder_filters[2], self.batchnorm, self.l2_reg, name='upsample_12')(conv2_1)
            cat1_2 = Concatenate(name='cat_12')([up1_2, conv1_1, encoder_outputs[1]])
            conv1_2 = ConvBlock(self.decoder_filters[2], batchnorm=self.batchnorm, l2_reg=self.l2_reg, name='conv_block_12')(cat1_2)

            up2_2 = UpsampleUnit(self.decoder_filters[1], self.batchnorm, self.l2_reg, name='upsample_22')(conv3_1)
            cat2_2 = Concatenate(name='cat_22')([up2_2, conv2_1, encoder_outputs[2]])
            conv2_2 = ConvBlock(self.decoder_filters[1], batchnorm=self.batchnorm, l2_reg=self.l2_reg, name='conv_block_22')(cat2_2)

            # column 3
            up0_3 = UpsampleUnit(self.decoder_filters[3], self.batchnorm, self.l2_reg, name='upsample_03')(conv1_2)
            cat0_3 = Concatenate(name='cat_03')([up0_3, conv0_2, conv0_1, encoder_outputs[0]])
            conv0_3 = ConvBlock(self.decoder_filters[3], batchnorm=self.batchnorm, l2_reg=self.l2_reg, name='conv_block_03')(cat0_3)

            up1_3 = UpsampleUnit(self.decoder_filters[2], self.batchnorm, self.l2_reg, name='upsample_13')(conv2_2)
            cat1_3 = Concatenate(name='cat_13')([up1_3, conv1_2, conv1_1, encoder_outputs[1]])
            conv1_3 = ConvBlock(self.decoder_filters[2], batchnorm=self.batchnorm, l2_reg=self.l2_reg, name='conv_block_13')(cat1_3)
            
            # column 3
            up0_4 = UpsampleUnit(self.decoder_filters[3], self.batchnorm, self.l2_reg, name='upsample_04')(conv1_3)
            cat0_4 = Concatenate(name='cat_04')([up0_4, conv0_3, conv0_2, conv0_1, encoder_outputs[0]])
            conv0_4 = ConvBlock(self.decoder_filters[3], batchnorm=self.batchnorm, l2_reg=self.l2_reg, name='conv_block_04')(cat0_4)

            if self.encoder_name == 'EfficientNetB7':
                up_out = UpsampleUnit(self.decoder_filters[4], self.batchnorm, self.l2_reg, name='upsample_out')(conv0_4)
                last_unit = up_out
            else:
                last_unit = conv0_4

            output = Conv2D(1, 1, activation='sigmoid', name='output', kernel_initializer='he_normal',
                                        padding='same', kernel_regularizer=l2(self.l2_reg))(last_unit)
            
            self.decoder = keras.Model(inputs=self.encoder.input, outputs=output, name='UNet++')


        else:
            raise ValueError('{} is not a valid decoder name'.format(self.decoder_name))
    
    def call(self):
        return self.decoder()