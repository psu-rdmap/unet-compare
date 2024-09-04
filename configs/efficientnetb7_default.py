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