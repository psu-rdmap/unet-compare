# example dict w/ default u-net backbone
encoder_dict = dict(
    name = 'EfficientNetB7',
    encoder_filters = (64, 48, 80, 224, 640), #(32, 64, 128, 256, 512),
    encoder_stages = ('stem_activation', 'block2g_add', 'block3g_add', 'block5j_add', 'block7d_add'), #('relu_00b', 'relu_10b', 'relu_20b', 'relu_30b', 'relu_40b')
)

decoder_dict = dict(
    l2_reg = 1e-3,
    batchnorm = True,
    decoder_filters = (256, 128, 64, 32, 16)
)

dataset_dict = dict(
    image_shape = (1024, 1024, 3),
)

encoder_dict.update(dict(l2_reg = decoder_dict['l2_reg']))
encoder_dict.update(dict(batchnorm = decoder_dict['batchnorm']))
encoder_dict.update(dict(image_shape = dataset_dict['image_shape']))

input_dict = dict(
    encoder_dict = encoder_dict,
    decoder_dict = decoder_dict,
    dataset_dict = dataset_dict
)