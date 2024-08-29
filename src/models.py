import keras
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Concatenate
from keras.regularizers import l2
from blocks import conv_block, upsample_unit


class Encoder:
    def __init__(self, input, encoder_dict):
        self.input = input
        self.encoder_name = encoder_dict['name']
        self.input_shape = encoder_dict['image_shape']
        self.encoder_stages = encoder_dict['encoder_stages']

        if self.encoder_name == 'default':            
            self.num_filters = encoder_dict['encoder_filters']
            self.l2_reg = encoder_dict['l2_reg']
            self.batchnorm = encoder_dict['batchnorm']

            self.encoder = Encoder.unet_default_encoder(self)
            
        elif self.encoder_name == 'EfficientNetB7':
            self.encoder = Encoder.efficientnet(self)

        else:
            raise Exception('{} is not a valid encoder name'.format(self.encoder_name))
        
        self.encoder_maps = Encoder.get_encoder_maps(self)
        
    def unet_default_encoder(self):
        
        conv0_0 = conv_block(self.input, '00',  self.num_filters[0], batchnorm=True, l2_reg=self.l2_reg)
        pool0 = MaxPooling2D(pool_size=(2, 2), name='pool0')(conv0_0)

        conv1_0 = conv_block(pool0, '10', self.num_filters[1], batchnorm=self.batchnorm, l2_reg=self.l2_reg)
        pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1_0)

        conv2_0 = conv_block(pool1, '20', self.num_filters[2], batchnorm=self.batchnorm, l2_reg=self.l2_reg)
        pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2_0)

        conv3_0 = conv_block(pool2, '30', self.num_filters[3], batchnorm=self.batchnorm, l2_reg=self.l2_reg)
        pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3_0)

        conv4_0 = conv_block(pool3, '40', self.num_filters[4], batchnorm=self.batchnorm, l2_reg=self.l2_reg)

        return keras.Model(inputs=self.input, outputs=conv4_0, name='UNet Encoder')

    def efficientnet(self):
        return tf.keras.applications.EfficientNetB7(
            include_top  = False,
            weights      = 'imagenet',
            input_tensor = self.input,
            input_shape  = self.input_shape,
        )
    
    def get_encoder_maps(self):
        outputs_dict = {layer: self.encoder.get_layer(layer).output for layer in self.encoder_stages}
        encoder = keras.Model(inputs=self.input, outputs=outputs_dict)
        encoder_outputs = encoder(self.input)
        return [encoder_outputs[stage] for stage in self.encoder_stages]


class Decoder:
    def __init__(self, encoder_maps, decoder_dict):
        self.encoder_maps = encoder_maps
        self.num_filters = decoder_dict['decoder_filters']
        self.l2_reg = decoder_dict['l2_reg']
        self.batchnorm = decoder_dict['batchnorm']

        self.decoder = Decoder.UNetDecoder(self)


    def UNetDecoder(self):
        up3_1 = upsample_unit(self.encoder_maps[4], '31', self.num_filters[0], batchnorm=self.batchnorm, l2_reg=self.l2_reg)
        cat3_1 = Concatenate(name='cat_31')([up3_1, self.encoder_maps[3]])
        conv3_1 = conv_block(cat3_1, '31', self.num_filters[0], batchnorm=self.batchnorm, l2_reg=self.l2_reg)
        
        up2_2 = upsample_unit(conv3_1, '22', self.num_filters[0], batchnorm=self.batchnorm, l2_reg=self.l2_reg)
        cat2_2 = Concatenate(name='cat_22')([up2_2, self.encoder_maps[2]])
        conv2_2 = conv_block(cat2_2, '22', self.num_filters[1], batchnorm=self.batchnorm, l2_reg=self.l2_reg)

        up1_3 = upsample_unit(conv2_2, '13', self.num_filters[1], batchnorm=self.batchnorm, l2_reg=self.l2_reg)
        cat1_3 = Concatenate(name='cat_13')([up1_3, self.encoder_maps[1]])
        conv1_3 = conv_block(cat1_3, '13', self.num_filters[2], batchnorm=self.batchnorm, l2_reg=self.l2_reg)

        up0_4 = upsample_unit(conv1_3, '04', self.num_filters[2], batchnorm=self.batchnorm, l2_reg=self.l2_reg)
        cat0_4 = Concatenate(name='cat_04')([up0_4, self.encoder_maps[0]])
        conv0_4 = conv_block(cat0_4, '04', self.num_filters[3], batchnorm=self.batchnorm, l2_reg=self.l2_reg)

        up_out = upsample_unit(conv0_4, '_out', self.num_filters[4], batchnorm=self.batchnorm, l2_reg=self.l2_reg)
            
        output = Conv2D(1, 1, activation='sigmoid', name='output', kernel_initializer='he_normal',
                                    padding='same', kernel_regularizer=l2(self.l2_reg))(up_out)

        return keras.Model(inputs=self.encoder_maps, outputs=output)
    

class UNet:
    def __init__(self, input_dict):
        self.encoder_dict = input_dict['encoder_dict']
        self.decoder_dict = input_dict['decoder_dict']
    
        self.model = UNet.builder(self)

    def builder(self):        
        input_shape = self.encoder_dict['image_shape']
        input_tensor = keras.Input(shape=input_shape, name='main_input')

        encoder = Encoder(input_tensor, self.encoder_dict)
        decoder = Decoder(encoder.encoder_maps, self.decoder_dict)

        return keras.Model(inputs=input_tensor, outputs=decoder.decoder.output)