import models
import dataloader
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from os.path import join
from os import mkdir
import time

def default(configs):
    
    # dataset
    print('\nCreating dataset...')
    time.sleep(1.5)
    dataset, train_steps, val_steps = dataloader.create_dataset(configs)
    print('\nTraining images: ', configs['train'])
    print('Validation images:', configs['val'])


    # model
    print('\nFetching and compiling model...')
    time.sleep(0.5)
    model = models.Decoder(configs).decoder
    
    # optimizer and loss
    model.compile(
        optimizer = Adam(learning_rate=configs['learning_rate']),
        loss = 'binary_crossentropy',
        metrics = ['accuracy', 'Precision', 'Recall']
    )
    
    # create directory to contain results
    mkdir(join(configs['root'], configs['results']))
    
    # training callbacks
    callbacks = [
        CSVLogger(
            join(configs['root'], configs['results'], 'metrics.csv'), 
            separator=',', 
            append=False
        ),
        ModelCheckpoint(
            join(configs['root'], configs['results'], 'best.weights.h5'), 
            verbose=1, 
            save_best_only=True, 
            save_weights_only=True
        )
    ]

    print('\nStarting training loop...\n')
    # start training 
    model.fit(
        dataset['train'],
        epochs=configs['num_epochs'],
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        validation_data=dataset['val'],
        callbacks=callbacks,
        verbose=1 # 1 = live progress bar, 2 = one line per epoch
    )

    # plot results
    

    # cleanup



def cross_val(configs):
    raise NotImplementedError
