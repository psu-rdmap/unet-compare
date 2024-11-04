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
    """
    This represents the encoder of the UNet Model. Either U-Net or EfficientNetB7 can be used.

    Parameters
    ----------
    configs : dict
        Input configs provided by the user
    
    Returns
    -------
    outputs (x) : list of tf.Tensors
        Encoder output at various stages
        Will be used for skip/dense connections in the decoder
    """
    
    def __init__(self, configs : dict):
        super().__init__()

        # unpack configs into attributes
        for key, value in configs.items():
            setattr(self, key, value)
        
        if self.encoder_name == 'UNet':
            self.encoder_stages = ["conv_block_00", "conv_block_10", "conv_block_20", "conv_block_30", "conv_block_40"]
            # define layers at each row (only 1 node per row)
            self._layers = []
            for idx, filters in enumerate(self.encoder_filters):
                name_idx = f'{idx}0'
                # add conv layer
                self._layers.append(ConvBlock(filters, self.batchnorm, self.l2_reg, name = 'conv_block_'+name_idx))
                # add pool layer
                self._layers.append(MaxPooling2D(pool_size=2, name = 'pool_'+name_idx))
            # remove last pool layer
            self._layers.pop(-1)
        elif self.encoder_name == 'EfficientNetB7':
            self.encoder_stages = ["stem_activation", "block2g_add", "block3g_add", "block5j_add", "block7d_add"]
            backbone = EfficientNetB7(
                include_top  = False,
                weights      = 'imagenet',
                input_shape  = self.input_shape)
            self.backbone = self.load_pretrained_backbone(backbone) 

    def load_pretrained_backbone(self, backbone):
        """
        Instantiate backbone by connecting inputs and outputs
        """
        backbone_outputs = [backbone.get_layer(stage).output for stage in self.encoder_stages]
        return tf.keras.Model(inputs = backbone.input, outputs = backbone_outputs)
      
    def call(self, inputs):
        x = inputs
        if self.encoder_name == 'UNet':
            outputs = {}
            for _layer in self._layers:
                x = _layer(x)
                outputs.update({_layer.name : x})
            return [outputs[stage] for stage in self.encoder_stages]
        elif self.encoder_name == 'EfficientNetB7':
            return self.backbone(x)


class Decoder(tf.keras.Model):
    """
    This represents the decoder of the UNet Model. Either U-Net or U-Net++ can be used.

    Parameters
    ----------
    configs : dict
        Input configs provided by the user
    
    Returns
    -------
    output (x) : tf.Tensor
        Final prediction of the model
        Designates end of forward pass
    """

    def __init__(self, configs : dict):
        super().__init__()

        # unpack configs into attributes
        for key, value in configs.items():
            setattr(self, key, value)

        # decoder specific layer orders
        self.rows = []
        if self.decoder_name == 'UNet':
            # only need one set of layers per row (1 node per layer)
            for row in range(4):
                name_idx = f'{3-row}{row+1}'
                filters = self.decoder_filters[row]
                _layers = [UpsampleUnit(filters, self.batchnorm, self.l2_reg, name='upsample_'+name_idx),
                          Concatenate(name='cat_'+name_idx),
                          ConvBlock(filters, batchnorm=self.batchnorm, l2_reg=self.l2_reg, name='conv_block_'+name_idx)]
                self.rows.append(_layers)
        elif self.decoder_name == 'UNet++':
            # each row has a different number of nodes and each node has an upsample, concatenate, and convolution layer
            for row in range(4):
                nodes = []
                for node in range(row+1):
                    name_idx = f'{3-row}{node+1}'
                    filters = self.decoder_filters[row]
                    _layers = [UpsampleUnit(filters, self.batchnorm, self.l2_reg, name='upsample_'+name_idx),
                              Concatenate(name='cat_'+name_idx),
                              ConvBlock(filters, batchnorm=self.batchnorm, l2_reg=self.l2_reg, name='conv_block_'+name_idx)]
                    nodes.append(_layers)
                self.rows.append(nodes)

        # final layers
        if self.encoder_name == 'EfficientNetB7':
            self.up_out = UpsampleUnit(self.decoder_filters[4], self.batchnorm, self.l2_reg, name='upsample_out')
        self.sigmoid = Conv2D(1, 1, activation='sigmoid', name='output', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(self.l2_reg))

    def call(self, enc_outputs):
        # starting from head of encoder
        enc_outputs.reverse()

        if self.decoder_name == 'UNet':
            z = enc_outputs[0]
            for row, _layers in enumerate(self.rows):
                x = _layers[0](z) # upsample
                y = _layers[1]([x, enc_outputs[row+1]]) # concatenate
                z = _layers[2](y) # convolution
        elif self.decoder_name == 'UNet++':
            # update previous row outs with previous encoder out and convolutions (for upsampling)
            prev_row_outs = [enc_outputs[0]]
            for row, nodes in enumerate(self.rows):
                # update current row outs with current encoder out and convolutions from the left (for concatenation)
                current_row_outs = [enc_outputs[row+1]]
                for node, _layers in enumerate(nodes):
                    x = _layers[0](prev_row_outs[node]) # upsample
                    y = _layers[1]([x] + current_row_outs[:(node+1)]) # concatenate
                    z = _layers[2](y) # convolution
                    current_row_outs.append(z)
                prev_row_outs = current_row_outs # reset prev row outs
            z = current_row_outs[-1]

        if self.encoder_name == 'EfficientNetB7':
            # need an extra upsample due to EfficientNet stem halving resolution
            z = self.up_out(z)
        return self.sigmoid(z)


class UNet(tf.keras.Model):
    """
    This is the fulle U-Net model with the encoder and decoder connected.

    Parameters
    ----------
    configs : dict
        Input configs provided by the user
    
    Returns
    -------
    output (x) : tf.Tensor
        Final prediction of the model from decoder
    """
    
    def __init__(self, configs : dict):
        super().__init__()

        for key, value in configs.items():
            setattr(self, key, value)
        
        # instantiate encoder and decoder
        self.encoder = Encoder(configs)
        self.decoder = Decoder(configs)

    def call(self, inputs):
        x = self.encoder(inputs)            
        return self.decoder(x)