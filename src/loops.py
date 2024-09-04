import models
import dataloader
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint


def default(configs):
    # dataset
    dataset, STEPS_PER_EPOCH, VALIDATION_STEPS = dataloader.create_dataset(configs)

    # model
    model = models.Decoder(**configs)

    # optimizer and loss
    model.compile(
        optimizer = Adam(learning_rate=configs['learning_rate']),
        loss = 'binary_crossentropy',
        metrics = ['accuracy', 'Precision', 'Recall']
    )

    # training
    callbacks = [
        CSVLogger(
            join(results_dir, "metrics.csv"), 
            separator=',', 
            append=False
        ),
        ModelCheckpoint(
            join(results_dir, "best_model_unet.h5"), 
            verbose=1, 
            save_best_only=True, 
            save_weights_only=True
        )
    ]

    model.fit(
        dataset['train'],
        epochs=configs['num_epochs'],
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        validation_data=dataset['val'],
        callbacks=callbacks,
        verbose=2
    )

    # results
    #plot_results()


    # cleanup


def cross_val(configs):
    raise NotImplementedError