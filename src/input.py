default_configs = dict(
    # general 
    l2_reg = 1e-3,
    batchnorm = True,
    image_shape = (1024, 1024, 3),
    # encoder
    encoder_name = 'default',
    encoder_filters = (32, 64, 128, 256, 512), # only required for vanilla encoder
    encoder_stages = ('conv_block_00', 'conv_block_10', 'conv_block_20', 'conv_block_30', 'conv_block_40'),
    # decoder
    decoder_filters = (512, 256, 128, 64, 32)
)   

efficientnet_configs = dict(
    # general 
    l2_reg = 1e-3,
    batchnorm = True,
    image_shape = (1024, 1024, 3),
    # encoder 
    encoder_name = 'EfficientNetB7',
    encoder_filters = (64, 48, 80, 224, 640),
    encoder_stages = ('stem_activation', 'block2g_add', 'block3g_add', 'block5j_add', 'block7d_add'),
    # decoder 
    decoder_filters = (512, 256, 128, 64, 32)
)