import argparse
import os, sys
from os import listdir, mkdir, remove
from os.path import join, splitext, exists, isdir, split
import datetime
from dataloader import dataloader
import tensorflow as tf
from keras.models import Model
import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from utils import get_model, copy_files, augment, parse_input
import re
import shutil
import numpy as np
from keras.metrics import F1Score
from keras import backend as K
import gc
from imageio import imread
from keras.preprocessing.image import save_img
import csv
from matplotlib import pyplot as plt
import os

# -----------Obtaining User Specifications-----------

parser = argparse.ArgumentParser(description='U-Net Training')

parser.add_argument('--dataset', default='dataset.dat', type=str,
                    help='Path to file containing dataset information')
parser.add_argument('--training', default='training.dat', type=str,
                    help='Path to file containing training information')

args = parser.parse_args()

# ---------------------------------------------------

# suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYCARET_CUSTOM_LOGGING_LEVEL'] = 'CRITICAL'

# check if gpu is available
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# get dataset and training info
tr_info, ds_fns = parse_input(args.training, args.dataset)

# add root to PATH environment variable
sys.path.append(tr_info['root'])

schedulers = ('step', 'misc')

def main():
    images_path = join(tr_info['root'], tr_info['images_path'])
    labels_path = join(tr_info['root'], tr_info['labels_path'])
    
    cv_toggle = eval(tr_info['cross_validation'])
    num_folds = eval(tr_info['num_folds'])
    restart_fold = eval(tr_info['restart_fold'])
    scheduler = tr_info['learning_scheduler']
    if scheduler in schedulers:
        schedule_toggle = True
    elif scheduler == 'False' or scheduler == 'None':
        schedule_toggle = False
    else:
        raise Exception(f'Learning rate scheduler {scheduler} not found')
    early_stopping = eval(tr_info['early_stopping'])
    if early_stopping:
        es_toggle = True
        if cv_toggle:
            raise Exception("Early stopping must be false if cross validation is activated")
    else:
        es_toggle = False


    # confirm user-specified information
    print("Model:", tr_info['model'])
    print("LR:", tr_info['learning_rate'])
    print("L2:", tr_info['l2_regularization'])
    print("Batch Size:", tr_info['batch_size'])
    print("Batchnorm:", tr_info['batchnorm'])
    print("Max Epochs:", tr_info['max_epochs'])
    if schedule_toggle:
        if scheduler == schedulers[0]:
            print("Scheduler: {} w/ decay {} per {} epochs".format(scheduler, tr_info['decay_fact'], tr_info['decay_freq']))
    print("Cross Validation:", tr_info['cross_validation'])
    if cv_toggle:
        print("Number of Folds:", tr_info['num_folds'])
        print("Restart at Fold:", tr_info['restart_fold'])
    print("Dataset:", tr_info['dataset_identifier'])
    print("Augment:", tr_info['augment'])
    if es_toggle:
        print("Early Stopping: {} at {} epochs".format(early_stopping, tr_info['patience']))
    else:
        print("Early Stopping:", early_stopping)
    print()
    
    # create image/label train set lists and verify that every image exists in the provided paths
    image_train_fns = [(fn + '.' + tr_info['images_ext']) for fn in ds_fns['train']]
    label_train_fns = [re.sub(tr_info['images_ext'], tr_info['labels_ext'], im) for im in image_train_fns]
    check_filenames(image_train_fns, images_path, tr_info['images_ext'])
    check_filenames(label_train_fns, labels_path, tr_info['labels_ext'])
    
    # create val lists if not using cross validation
    if not cv_toggle:
        image_val_fns = [(fn + '.' + tr_info['images_ext']) for fn in ds_fns['val']]
        label_val_fns = [re.sub(tr_info['images_ext'], tr_info['labels_ext'], im) for im in image_val_fns]
        check_filenames(image_val_fns, images_path, tr_info['images_ext'])
        check_filenames(label_val_fns, labels_path, tr_info['labels_ext'])
    
    loocv_toggle = False
    # determine number of images belonging to each fold
    if cv_toggle:
        assert num_folds > 0 and type(num_folds) == int, 'Number of folds must be a positive, non-zero integer'
        num_images_per_fold = int(np.floor(len(image_train_fns)/num_folds))
        print('Number of images in dataset:', len(image_train_fns))
        print('Number of images per fold:', num_images_per_fold)
        assert num_images_per_fold > 0, 'There cannot be more folds than images'
        if num_folds == len(image_train_fns):
            loocv_toggle = True
            print("LOOCV detected; training time may be signficant")
        else:
            loocv_toggle = False
    
    # define latent dataset path
    latent_ds_path = join(tr_info['root'], 'latent_ds')
   
    # remove a previous latent directory if it exists
    if isdir(latent_ds_path):
        shutil.rmtree(latent_ds_path)

    # restart operations for CV
    if restart_fold:
        print('\nRestarting cross validation at fold index {}\n'.format(tr_info['restart_fold']))
    
    # for non LOOCV shuffle to prevent sequence bias
    if cv_toggle and not loocv_toggle:
        SEED = 401
        np.random.seed(SEED)
        image_train_fns_shuffled = np.ndarray.tolist(np.random.permutation(image_train_fns.copy()))
   
    # create directory for results with an extra token for cv info
    if loocv_toggle:
        cv_string = '_loocv'
    elif cv_toggle: 
        cv_string = '_cv' + str(num_folds)
    else:
        cv_string = ''
    
    # find directory existing directory is we are restarting
    if not cv_toggle:
        # CV off
        now = datetime.datetime.now()
        results_dir = "results_" + tr_info['dataset_identifier'] + cv_string + '_' + tr_info['model'] + now.strftime("_(%Y-%m-%d)_(%H-%M-%S)")
        results_dir = join(tr_info['root'], results_dir)
        mkdir(results_dir)
    elif cv_toggle and not restart_fold:
        # CV on, not restarting
        now = datetime.datetime.now()
        results_dir = "results_" + tr_info['dataset_identifier'] + cv_string + '_' + tr_info['model'] + now.strftime("_(%Y-%m-%d)_(%H-%M-%S)")
        results_dir = join(tr_info['root'], results_dir)
        mkdir(results_dir)
    else:
        # CV on, restarting
        assert isdir(join(tr_info['root'], tr_info['results_dir_path']))
        results_dir = join(tr_info['root'], tr_info['results_dir_path'])
    
    # set training start/stopping indices
    start_fold = restart_fold
    if not cv_toggle:
        stop_fold = 1
    else:
        stop_fold = num_folds
    
    # store metrics for each fold
    if cv_toggle:
        # set array dimensions (num epochs x num folds)
        c_dim = eval(tr_info['max_epochs'])
        r_dim = eval(tr_info['num_folds'])
        
        # initialize arrays
        train_loss = np.ones((r_dim, c_dim))
        train_precision = np.zeros((r_dim, c_dim))
        train_recall = np.zeros((r_dim, c_dim))
        train_f1 = np.zeros((r_dim, c_dim))
        
        val_loss = np.ones((r_dim, c_dim))
        val_precision = np.zeros((r_dim, c_dim))
        val_recall = np.zeros((r_dim, c_dim))
        val_f1 = np.zeros((r_dim, c_dim))
    

    # start training
    for fold_idx in range(start_fold, stop_fold):
        # perform cv operations
        if cv_toggle:
            print('\n' + '-'*40 + f' Fold {fold_idx} ' + '-'*40 + '\n')   

            # LOOCV splitting dataset
            if loocv_toggle:
                hold_out_image = image_train_fns[fold_idx]
                print('Current hold out image:', hold_out_image)

                # define latent train set
                latent_image_train_set = image_train_fns.copy()
                
                # remove the hold out image from the list
                latent_image_train_set.remove(hold_out_image)

                # add it to the validation set
                latent_image_val_set = [hold_out_image]
            
            # normal cv splitting
            else:
                low_idx = fold_idx*num_images_per_fold
            
                # final fold will include remainder images
                if fold_idx == num_folds - 1:
                    latent_image_val_set = image_train_fns_shuffled[low_idx:]
                else:
                    high_idx = (fold_idx+1)*num_images_per_fold
                    latent_image_val_set = image_train_fns_shuffled[low_idx:high_idx]
            
                print('Current hold out images:', ', '.join(map(str, latent_image_val_set)))
            
                # define latent train set
                latent_image_train_set = image_train_fns_shuffled.copy()
                
                # remove hold out images from train set
                for hold_out_image in latent_image_val_set:
                    latent_image_train_set.remove(hold_out_image)
        
        # normal training
        else:
            latent_image_train_set = image_train_fns.copy()
            latent_image_val_set = image_val_fns.copy()
            
            print('Current training images:', ', '.join(map(str, latent_image_train_set)))
            print('Current validation images:', ', '.join(map(str, latent_image_val_set)))
        
        # create the latent dataset
        print("\nCreating the dataset...")
        create_latent_dataset(latent_ds_path, latent_image_train_set, latent_image_val_set, images_path, labels_path, tr_info['images_ext'], tr_info['labels_ext'], augment_toggle = eval(tr_info['augment']))
        
        # batch the latent dataset and return some training info
        dataset, STEPS_PER_EPOCH, VALIDATION_STEPS = dataloader(latent_ds_path, eval(tr_info['batch_size']), tr_info['images_ext'], tr_info['labels_ext'])

        # load model
        model = get_model(tr_info['model'], eval(tr_info['image_size']), eval(tr_info['l2_regularization']), eval(tr_info['batchnorm']))

        # compile model
        model.compile(optimizer=Adam(learning_rate=eval(tr_info['learning_rate'])), loss = 'binary_crossentropy',
                      metrics= ['accuracy', 'Precision', 'Recall'])
        
        # create additional results structure for cv
        if cv_toggle: 
            if loocv_toggle:
                fold_results_dir = join(results_dir, splitext(image_train_fns[fold_idx])[0])
            else:
                fold_results_dir = join(results_dir, str(fold_idx))
            
            # in case we are restarting and this fold dir already exists
            if isdir(fold_results_dir):
                shutil.rmtree(fold_results_dir)
            
            mkdir(fold_results_dir)
        else:
            fold_results_dir = results_dir
        
        # define callbacks for saving results and early stopping
        callbacks = [CSVLogger(join(fold_results_dir, "metrics.csv"), separator=',', append=False),
                     ModelCheckpoint(join(fold_results_dir, "best_model_unet.h5"), verbose=1, save_best_only=True, save_weights_only=True)]
        
        # add the listed scheduler    
        if schedule_toggle:
            if scheduler == schedulers[0]:
                callbacks.append(LearningRateScheduler(step_schedule))
    
        # add early stopping mechanism if listed
        if es_toggle:
            callbacks.append(EarlyStopping(patience = eval(tr_info['patience'])))
        
        print("\nStarting training...")

        # train with latent dataset
        model_history = model.fit(dataset['train'],
                                  epochs=eval(tr_info['max_epochs']),
                                  steps_per_epoch=STEPS_PER_EPOCH,
                                  validation_steps=VALIDATION_STEPS,
                                  validation_data=dataset['val'],
                                  callbacks=callbacks,
                                  verbose=2)
        
        print("\nPlotting results...")

        # plot metrics for fold
        fold_metrics = plot_fold_metrics(join(fold_results_dir, "metrics.csv"))
        
        # store metrics for fold
        if cv_toggle:
            train_loss[fold_idx, :] = fold_metrics['t_loss']
            train_precision[fold_idx, :] = fold_metrics['t_precision']
            train_recall[fold_idx, :] = fold_metrics['t_recall']
            train_f1[fold_idx, :] = fold_metrics['t_f1']
            
            val_loss[fold_idx, :] = fold_metrics['v_loss']
            val_precision[fold_idx, :] = fold_metrics['v_precision']
            val_recall[fold_idx, :] = fold_metrics['v_recall']
            val_f1[fold_idx, :] = fold_metrics['v_f1']

        # inference all images if cv is not activated
        if not cv_toggle:
            print("\nSaving model predictions on training set")
            save_dir = join(results_dir, 'train_predictions')
            mkdir(save_dir)
            
            for fn in image_train_fns:
                inference(images_path, tr_info['images_ext'], fn, model, save_dir)
            
            print("\nSaving model predictions on validation set")
            save_dir = join(results_dir, 'val_predictions')
            mkdir(save_dir)

            for fn in image_val_fns:
                inference(images_path, tr_info['images_ext'], fn, model, save_dir)

        # inference current validation images for fold
        if cv_toggle:
            print("\nSaving model predictions on fold validation set")
            save_dir = join(results_dir, 'val_predictions')
            if not isdir(save_dir):
                mkdir(save_dir)

            for fn in latent_image_val_set:
                inference(images_path, tr_info['images_ext'], fn, model, save_dir)
       
        # remove the current latent dataset
        shutil.rmtree(latent_ds_path)
        
        # clear GPU memory, collect garbage
        K.clear_session()
        gc.collect()
     
    # combine loss plots and average metrics if cv is toggled
    if cv_toggle:
        all_metrics_dict = dict(t_loss = train_loss,
                                t_precision = train_precision,
                                t_recall = train_recall,
                                t_f1 = train_f1,
                                v_loss = val_loss,
                                v_precision = val_precision,
                                v_recall = val_recall,
                                v_f1 = val_f1)
        
        plot_cv_loss(results_dir, all_metrics_dict, eval(tr_info['max_epochs']), num_folds)
        plot_cv_metrics(results_dir, all_metrics_dict, eval(tr_info['max_epochs']), num_folds)
    
    # copy current training/dataset files to results directory
    shutil.copyfile(args.dataset, join(results_dir, 'dataset.dat'))
    shutil.copyfile(args.training, join(results_dir, 'training.dat'))


def inference(path_to, img_ext, fn, model, out_dir):
    # open each image as numpy array and inference it
    fn_path = join(path_to, fn)
    img = tf.io.read_file(fn_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img,axis=0)
    pred = model.predict(img, verbose = 2)
    out_path = join(out_dir, re.sub(img_ext, 'png', fn))
    save_img(out_path, np.squeeze(pred, axis=0))

def plot_cv_loss(results_dir, metrics, num_epochs, num_folds):
    # create the epoch iterator
    epoch = range(num_epochs)
    
    # initialize average metrics
    mean_train_loss = np.ones(num_epochs)
    mean_val_loss = np.ones(num_epochs)

    # average across folds
    for ep in epoch:
        mean_train_loss[ep] = np.mean(metrics['t_loss'][:, ep])
        mean_val_loss[ep] = np.mean(metrics['v_loss'][:, ep])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,17))
    
    # plot all loss curves together
    for fold in range(num_folds):
        ax1.plot(epoch, metrics['t_loss'][fold, :], label='Fold {}'.format(fold))
        ax2.plot(epoch, metrics['v_loss'][fold, :], label='Fold {}'.format(fold))
        
    # plot averaged loss curves
    ax3.plot(epoch, mean_train_loss, '-o',  label='Train')
    ax3.plot(epoch, mean_val_loss, '-o', label='Val')

    # figure specs
    ax1.legend()
    ax2.legend()
    ax3.legend()

    ax1.set_xlim([0,num_epochs])
    ax2.set_xlim([0,num_epochs])
    ax3.set_xlim([0,num_epochs])

    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')

    ax1.set_xlabel('Epoch')
    ax2.set_xlabel('Epoch')
    ax3.set_xlabel('Epoch')

    ax1.set_ylabel('Train Loss')
    ax2.set_ylabel('Validation Loss')
    ax3.set_ylabel('Averaged Loss')

    ax1.grid(visible=True)
    ax2.grid(visible=True)
    ax3.grid(visible=True)

    fig.savefig(join(results_dir, 'loss.png'), bbox_inches="tight")


def plot_cv_metrics(results_dir, metrics, num_epochs, num_folds):
    # create the epoch iterator
    epoch = range(num_epochs)
    
    # initialize average metrics
    mean_train_precision = np.zeros(num_epochs)
    mean_train_recall = np.zeros(num_epochs)
    mean_train_f1 = np.zeros(num_epochs)

    mean_val_precision = np.zeros(num_epochs)
    mean_val_recall = np.zeros(num_epochs)
    mean_val_f1 = np.zeros(num_epochs)

    # average across folds
    for ep in epoch:
        mean_train_precision[ep] = np.mean(metrics['t_precision'][:, ep])
        mean_train_recall[ep] = np.mean(metrics['t_recall'][:, ep])
        mean_train_f1[ep] = np.mean(metrics['t_f1'][:, ep])

        mean_val_precision[ep] = np.mean(metrics['v_precision'][:, ep])
        mean_val_recall[ep] = np.mean(metrics['v_recall'][:, ep])
        mean_val_f1[ep] = np.mean(metrics['v_f1'][:, ep])
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12,17))
    ax1.plot(epoch, mean_train_precision, '-o', label='Train')
    ax1.plot(epoch, mean_val_precision, '-o', label='Val')
    ax1.legend()
    ax1.set_xlim([0,num_epochs])
    ax1.set_ylim([0,1])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Precision')
    ax1.grid(visible=True)
    ax1.set_yticks(ticks=np.arange(0,1.1,0.1))

    ax2.plot(epoch, mean_train_recall, '-o', label='Train')
    ax2.plot(epoch, mean_val_recall, '-o', label='Val')
    ax2.legend()
    ax2.set_xlim([0,num_epochs])
    ax2.set_ylim([0,1])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Recall')
    ax2.grid(visible=True)
    ax2.set_yticks(ticks=np.arange(0,1.1,0.1))

    ax3.plot(epoch, mean_train_f1, '-o', label='Train')
    ax3.plot(epoch, mean_val_f1, '-o', label='Val')
    ax3.legend()
    ax3.set_xlim([0,num_epochs])
    ax3.set_ylim([0,1])
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1')
    ax3.grid(visible=True)
    ax3.set_yticks(ticks=np.arange(0,1.1,0.1))

    fig.savefig(join(results_dir, 'metrics.png'), bbox_inches="tight")


def plot_fold_metrics(metrics_path):
    def get_num_lines(path):
        # get number of epochs for storing metrics
        with open(path, mode='r', newline='') as csv_file:
            reader = csv.reader(csv_file)
            line_count = sum(1 for row in reader)
        return line_count
    
    num_epochs = get_num_lines(metrics_path) - 1
    epoch = range(num_epochs)

    # initialize arrays to store metrics
    train_loss = np.zeros(num_epochs)
    train_precision = np.zeros(num_epochs) 
    train_recall = np.zeros(num_epochs)
    train_f1 = np.zeros(num_epochs)

    val_loss = np.zeros(num_epochs)
    val_precision = np.zeros(num_epochs)
    val_recall = np.zeros(num_epochs)
    val_f1 = np.zeros(num_epochs)

    # get metrics
    with open(metrics_path, mode='r', newline='') as metrics:
        metrics_reader = csv.reader(metrics)
        next(metrics_reader)
        
        epsilon = 1e-7
        epoch_idx = 0
        for row in metrics_reader:
            train_loss[epoch_idx] = float(row[2])
            train_precision[epoch_idx] = float(row[3])
            train_recall[epoch_idx] = float(row[4])

            val_loss[epoch_idx] = float(row[6])
            val_precision[epoch_idx] = float(row[7])
            val_recall[epoch_idx] = float(row[8])

            train_f1[epoch_idx] = 2*float(row[3])*float(row[4])/(float(row[3])+float(row[4])+epsilon)
            val_f1[epoch_idx] = 2*float(row[7])*float(row[8])/(float(row[7]) + float(row[8])+epsilon)

            epoch_idx += 1   

    # get epoch corresponding to lowest loss
    best_idx, = np.where(np.isclose(val_loss, min(val_loss)))[0]

    fig, (ax1,ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12,17))

    ax1.plot(epoch, train_loss, '-o',  label='Train')
    ax1.plot(epoch, val_loss, '-o', label='Val')
    ax1.plot(best_idx, train_loss[best_idx], 'D', color='purple', label='Min Val Loss')
    ax1.plot(best_idx, val_loss[best_idx], 'D', color='purple')
    ax1.legend()
    ax1.set_xlim([0,num_epochs])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss')
    ax1.set_yscale('log')
    ax1.grid(visible=True)
  
    ax2.plot(epoch, train_precision, '-o', label='Train')
    ax2.plot(epoch, val_precision, '-o', label='Val')
    ax2.plot(best_idx, train_precision[best_idx], 'D', color='purple', label='Min Val Loss')
    ax2.plot(best_idx, val_precision[best_idx], 'D', color='purple')
    ax2.legend()
    ax2.set_xlim([0,num_epochs])
    ax2.set_ylim([0,1])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Precision')
    ax2.grid(visible=True)
    ax2.set_yticks(ticks=np.arange(0,1.1,0.1))

    ax3.plot(epoch, train_recall, '-o', label='Train')
    ax3.plot(epoch, val_recall, '-o', label='Val')
    ax3.plot(best_idx, train_recall[best_idx], 'D', color='purple', label='Min Val Loss')
    ax3.plot(best_idx, val_recall[best_idx], 'D', color='purple')
    ax3.legend()
    ax3.set_xlim([0,num_epochs])
    ax3.set_ylim([0,1])
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Recall')
    ax3.grid(visible=True)
    ax3.set_yticks(ticks=np.arange(0,1.1,0.1))

    ax4.plot(epoch, train_f1, '-o', label='Train')
    ax4.plot(epoch, val_f1, '-o', label='Val')
    ax4.plot(best_idx, train_f1[best_idx], 'D', color='purple', label='Min Val Loss')
    ax4.plot(best_idx, val_f1[best_idx], 'D', color='purple')
    print('\nVal F1-Score at Lowest Loss:', str(val_f1[best_idx]))
    ax4.legend()
    ax4.set_xlim([0,num_epochs])
    ax4.set_ylim([0,1])
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1')
    ax4.grid(visible=True)
    ax4.set_yticks(ticks=np.arange(0,1.1,0.1))

    fig.savefig(join(split(metrics_path)[0], 'metrics.png'), bbox_inches="tight")  
    
    fold_metrics_dict = dict(t_loss = train_loss,
                             t_precision = train_precision,
                             t_recall = train_recall,
                             t_f1 = train_f1,
                             v_loss = val_loss,
                             v_precision = val_precision,
                             v_recall = val_recall,
                             v_f1 = val_f1)

    return fold_metrics_dict

def step_schedule(epoch, lr):
    if epoch % eval(tr_info['decay_freq']) == 0 and epoch > 1:
        return lr * eval(tr_info['decay_fact'])
    else:
        return lr


def create_latent_dataset(destination, train_set, val_set, img_path, lab_path, img_ext, lab_ext, augment_toggle = False):    
    # create the directory tree of the latent dataset
    image_dir = join(destination,'images')
    image_train = join(image_dir, 'train')
    image_val = join(image_dir, 'val')

    annotation_dir = join(destination,'labels')
    annotation_train = join(annotation_dir, 'train')
    annotation_val = join(annotation_dir, 'val')

    mkdir(destination)

    mkdir(image_dir)
    mkdir(image_train)
    mkdir(image_val)

    mkdir(annotation_dir)
    mkdir(annotation_train)
    mkdir(annotation_val)

    # define image and label sets
    image_train_set = train_set
    image_val_set = val_set
    
    label_train_set = [re.sub(img_ext, lab_ext, im) for im in train_set]
    label_val_set = [re.sub(img_ext, lab_ext, im) for im in val_set]    

    # populate the latent dataset
    copy_files(image_train_set, img_path, image_train)
    copy_files(image_val_set, img_path, image_val)

    copy_files(label_train_set, lab_path, annotation_train)
    copy_files(label_val_set, lab_path, annotation_val)
    
    if augment_toggle:
        # augment the latent dataset
        augment(image_train, image_train_set, img_ext)
        augment(annotation_train, label_train_set, lab_ext)

def check_filenames(image_set, path, ext):
    for fn in image_set:
        if fn == '.' + ext:
            raise Exception("There is a null filename in the dataset input file!")
        elif fn not in listdir(path):
            raise Exception(f"File {fn} mentioned in the input file can not be found!")

if __name__ == '__main__':
    main()
