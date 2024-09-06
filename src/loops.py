import models
import dataloader
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from os.path import join


def default(configs):
    # dataset
    dataset, train_steps, val_steps = dataloader.create_dataset(configs)

    # model
    model = models.UNetModel(**configs)

    # optimizer and loss
    model.decoder.compile(
        optimizer = Adam(learning_rate=configs['learning_rate']),
        loss = 'binary_crossentropy',
        metrics = ['accuracy', 'Precision', 'Recall']
    )

    # training callbacks
    callbacks = [
        CSVLogger(
            join(configs['root'], configs['results_dir_path'], 'metrics.csv'), 
            separator=',', 
            append=False
        ),
        ModelCheckpoint(
            join(configs['root'], configs['results_dir_path'], 'best.weights.h5'), 
            verbose=1, 
            save_best_only=True, 
            save_weights_only=True
        )
    ]
    
    print("made it to fit")
    """
    # start training 
    model.fit(
        dataset['train'],
        epochs=configs['num_epochs'],
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        validation_data=dataset['val'],
        callbacks=callbacks,
        verbose=2 # 1 = live progress bar, 2 = one line per epoch
    )
    """

    # plot results

    # cleanup


def cross_val(configs):
    raise NotImplementedError