from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose
from keras.regularizers import l2

def conv_block(input_tensor, index, num_filters, kernel_size=3, batchnorm=True, l2_reg=1e-3):
    def conv_unit(input, unit_index):
        conv = Conv2D(num_filters, kernel_size, padding='same', name='conv_'+unit_index, use_bias=not(batchnorm), kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(input)
        if batchnorm:
            bn = BatchNormalization(name='bn_'+unit_index)(conv)
            act = Activation('relu', name='relu_'+unit_index)(bn)
        else:
            act = Activation('relu', name='relu_'+unit_index)(conv)

        return act
    
    conv1 = conv_unit(input_tensor, index + 'a')
    return conv_unit(conv1, index + 'b')



def upsample_unit(input_tensor, index, num_filters, kernel_size=2, batchnorm=True, l2_reg=1e-3):
    conv_up = Conv2DTranspose(num_filters, kernel_size, strides=(2,2), padding='same', name='up_'+index, use_bias=not(batchnorm), kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg))(input_tensor)
    if batchnorm:
        bn = BatchNormalization(name='bn'+index)(conv_up)
        act = Activation('relu', name='relu'+index)(bn)
    else:
        act = Activation('relu', name='relu'+index)(conv_up)
    
    return act