import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose
from keras.regularizers import l2


class ConvUnit(tf.keras.Layer):
    """
    Custom Keras layer that does Conv2D -> Batchnorm (optional) -> ReLU Activation
    """

    def __init__(self, num_filters, unit_index, batchnorm, l2_reg):
        super().__init__()
        self.num_filters = num_filters
        self.unit_index = unit_index
        self.batchnorm = batchnorm
        self.l2_reg = l2_reg
                
        self.conv = Conv2D(self.num_filters, 
                           3,
                           padding='same', 
                           name='conv_'+self.unit_index, 
                           use_bias=not(self.batchnorm), 
                           kernel_initializer='he_normal', 
                           kernel_regularizer=l2(self.l2_reg)
        )
        self.bn = BatchNormalization(name='bn_'+self.unit_index)
        self.act = Activation('relu', name='relu_'+self.unit_index)

    def call(self, inputs):
        conv = self.conv(inputs)
        if self.batchnorm:
            bn = self.bn(conv)
            return self.act(bn)
        else:
            return self.act(conv)


class ConvBlock(tf.keras.Layer):
    """
    Custom Keras layer that does two ConvUnits (Conv2D -> Batchnorm (optional) -> ReLU Activation)
    """

    def __init__(self, num_filters, batchnorm, l2_reg, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.index = kwargs.__getitem__('name')[-2:] # last two characters in name are always layer index
        self.batchnorm = batchnorm
        self.l2_reg = l2_reg

        self.conv_unit_1 = ConvUnit(self.num_filters, self.index+'a', self.batchnorm, self.l2_reg)
        self.conv_unit_2 = ConvUnit(self.num_filters, self.index+'b', self.batchnorm, self.l2_reg)

    def call(self, inputs):
        conv1 = self.conv_unit_1(inputs)
        return self.conv_unit_2(conv1)
    

class UpsampleUnit(tf.keras.Layer):
    """
    Custom Keras layer that does Conv2DTranspose (upsampling) -> Batchnorm (optional) -> ReLU Activation
    """
    
    def __init__(self, num_filters, batchnorm, l2_reg, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.index = kwargs.__getitem__('name')[-2:] # last two characters in name are always layer index
        self.batchnorm = batchnorm
        self.l2_reg = l2_reg
                
        self.conv_up = Conv2DTranspose(self.num_filters,
                                       2,
                                       padding='same', 
                                       name='up_'+self.index, 
                                       use_bias=not(self.batchnorm), 
                                       kernel_initializer='he_normal', 
                                       kernel_regularizer=l2(self.l2_reg),
                                       strides=2        
        )
        self.bn = BatchNormalization(name='bn_'+self.index)
        self.act = Activation('relu', name='relu_'+self.index)

    def call(self, inputs):
        conv_up = self.conv_up(inputs)
        if self.batchnorm:
            bn = self.bn(conv_up)
            return self.act(bn)
        else:
            return self.act(conv_up)