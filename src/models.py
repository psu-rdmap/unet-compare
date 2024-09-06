import keras
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Concatenate
from keras.regularizers import l2
from blocks import ConvBlock, UpsampleUnit


class Encoder:
    """
    Class that represents the encoder/backbone of the network

    Attributes
    ----------
    configs : dict
        dictionary containing all data necessary to build the encoder
    
    Methods
    -------
    unet_default_encoder
        Keras model built using Sequential that contains the architecture of the vanilla U-Net encoder
    efficientnet
        Keras functional model that contains the EfficientNetB7 feature extractor
    """

    def __init__(self, **kwargs):
        # pass in configs dict as kwargs and unpack into class attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # choose encoder
        if self.encoder_name == 'Vanilla':
            self.model = Encoder.unet_vanilla_encoder(self)
        elif self.encoder_name == 'EfficientNetB7':
            self.model = Encoder.efficientnet(self)
        else:
            raise Exception('{} is not a valid encoder name'.format(self.encoder_name))
    
    def unet_vanilla_encoder(self):
        return keras.Sequential(
            [                
                keras.Input(self.data['input_shape']),
                
                ConvBlock(self.encoder_filters[0], self.batchnorm, self.l2_reg, name = 'conv_block_00'),
                MaxPooling2D(pool_size=2, name='pool_0'),

                ConvBlock(self.encoder_filters[1], self.batchnorm, self.l2_reg, name = 'conv_block_10'),
                MaxPooling2D(pool_size=2, name='pool_1'),

                ConvBlock(self.encoder_filters[2], self.batchnorm, self.l2_reg, name = 'conv_block_20'),
                MaxPooling2D(pool_size=2, name='pool_2'),

                ConvBlock(self.encoder_filters[3], self.batchnorm, self.l2_reg, name = 'conv_block_30'),
                MaxPooling2D(pool_size=2, name='pool_3'),

                ConvBlock(self.encoder_filters[4], self.batchnorm, self.l2_reg, name = 'conv_block_40'),
            ],
            name = 'UNet Vanilla Encoder'
        )
        
    def efficientnet(self):
        return tf.keras.applications.EfficientNetB7(
            include_top  = False,
            weights      = 'imagenet',
            input_shape  = self.data['input_shape'],
        )
    

class UNetModel:
    """
    Class that represents full network

    Attributes
    ----------
    configs : dict
        dictionary containing all data necessary to build the decoder and attach it to the encoder
    
    Methods
    -------
    unet
        Keras model built using the Functional API that contains the architecture of the vanilla U-Net decoder
    get_encoder_outputs
        extract the feature maps from stages along the encoder for skip connections
    """

    def __init__(self, **kwargs):
        # pass in configs dict as kwargs and unpack into class attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # initialize encoder and decoder
        self.encoder = Encoder(**kwargs)
        self.decoder = UNetModel.unet(self)

    def unet(self):
        encoder_outputs = UNetModel.get_encoder_outputs(self)
    
        up3_1 = UpsampleUnit(self.decoder_filters[1], self.batchnorm, self.l2_reg, name='upsample_31')(encoder_outputs[4])
        cat3_1 = Concatenate(name='cat_31')([up3_1, encoder_outputs[3]])
        conv3_1 = ConvBlock(self.decoder_filters[1], batchnorm=self.batchnorm, l2_reg=self.l2_reg, name='conv_block_31')(cat3_1)
        
        up2_2 = UpsampleUnit(self.decoder_filters[2], self.batchnorm, self.l2_reg, name='upsample_22')(conv3_1)
        cat2_2 = Concatenate(name='cat_22')([up2_2, encoder_outputs[2]])
        conv2_2 = ConvBlock(self.decoder_filters[2], batchnorm=self.batchnorm, l2_reg=self.l2_reg, name='conv_block_22')(cat2_2)

        up1_3 = UpsampleUnit(self.decoder_filters[3], self.batchnorm, self.l2_reg, name='upsample_13')(conv2_2)
        cat1_3 = Concatenate(name='cat_13')([up1_3, encoder_outputs[1]])
        conv1_3 = ConvBlock(self.decoder_filters[3], batchnorm=self.batchnorm, l2_reg=self.l2_reg, name='conv_block_13')(cat1_3)

        up0_4 = UpsampleUnit(self.decoder_filters[4], self.batchnorm, self.l2_reg, name='upsample_04')(conv1_3)
        cat0_4 = Concatenate(name='cat_04')([up0_4, encoder_outputs[0]])
        conv0_4 = ConvBlock(self.decoder_filters[4], batchnorm=self.batchnorm, l2_reg=self.l2_reg, name='conv_block_04')(cat0_4)
            
        output = Conv2D(1, 1, activation='sigmoid', name='output', kernel_initializer='he_normal',
                                    padding='same', kernel_regularizer=l2(self.l2_reg))(conv0_4)

        return keras.Model(inputs=encoder_outputs, outputs=output, name='UNet')
    
    def get_encoder_outputs(self):
        # create Input object
        input = keras.Input(self.data['input_shape'])
        # get encoder outputs from specific layers (provided by encoder_stages tuple in configs)
        outputs_dict = {layer: self.encoder.model.get_layer(layer).output for layer in self.encoder_stages}
        # define encoder as Functional model with outputs corresponding to the pre-defined stages
        encoder_def = keras.Model(inputs=input, outputs=outputs_dict)
        # call encoder to create defined outputs
        encoder_outputs = encoder_def(input)
        # return list of feature map tensors from each stage
        return [encoder_outputs[stage] for stage in self.encoder_stages]