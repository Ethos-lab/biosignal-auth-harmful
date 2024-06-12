import argparse
import json
import numpy as np
import tensorflow as tf
import os
import loader
import util
from shutil import copyfile

from keras import backend as K

os.environ['CUDA_VISIBLE_DEVICES']='0,1' # can only use one 

def train(params):
    np.random.seed(2)
    print("Loading training set...")
    train = loader.prepare_dataset(params['train'], params)
    print("Loading val set...")
    val = loader.prepare_dataset(params['val'], params)

    print("Building preprocessor...")
    preproc = loader.Preproc(train)

    print("Train size: " + str(len(train['ecg'])) + " examples.")
    print("Val size: " + str(len(val['ecg'])) + " examples.")

    save_dir = util.make_save_dir(params['base_save_dir'], params['experiment'])
    copyfile(params['filename'], save_dir + '/' + params['filename'].replace('\\', '/').split('/')[-1])
    util.save(preproc, save_dir)

    if 'initial_weights' in params.keys() and len(params['initial_weights']) > 0:
        print('Loading initial weights from: ', params['initial_weights'])
        model = tf.keras.models.load_model(params['initial_weights'],custom_objects={"K": K}) # VEENA fixed this
        trainable = params.get("trainable", True)
        if not trainable:
            for l in model.layers:  l.trainable = False
        if params['experiment'].find('identification') >= 0:
            model = util.get_model_finetun_identification(model, params['lead_i'], params['individuals'])
        model = util.add_compile(model, **params)

    else:
        model = util.get_model(**params)

    stopping = tf.keras.callbacks.EarlyStopping(
        patience=params['patience_es'],
        min_delta=params['min_delta_es'],
        monitor=params['monitor'])

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=params['monitor'],
        min_delta=params['min_delta_rlr'],
        factor=params['factor'],
        patience=params['patience_rlr'],
        verbose=1,
        min_lr=params['min_lr'])

    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=util.get_filename_for_saving(save_dir, params['metrics']),
        save_best_only=params['save_best_only'])

    batch_size = params.get("batch_size", 16)

    if params['experiment'].find('verification') >= 0:
        train_gen, val_gen, train_len, val_len = util.verification_sample_creation(train, val, preproc, **params)
    else:
        input_keys_json = params['input_keys_json']
        output_keys_json = params['output_keys_json']
        if params['experiment'].find('identification') >= 0 or params['experiment'].find('finetun') >= 0:
            resample_max = params.get("resample_max", False)
            train = util.fix_data(train, resample_max)
            val = util.fix_data(val, resample_max)
        train_gen = loader.data_generator(batch_size, preproc, train, input_keys_json, output_keys_json)
        train_len = len(train['ecg'])
        val_gen = loader.data_generator(batch_size, preproc, val, input_keys_json, output_keys_json)
        val_len = len(val['ecg'])

    import math
    print("Len train examples: ", train_len, " - Len val examples: ", val_len, " - Batch size: ", batch_size, " - Steps per train epoch: ", math.ceil(train_len / batch_size), " - Steps per val epoch: ", math.ceil(val_len / batch_size), "v", int(val_len/batch_size))

    try:
        history = model.fit(
            train_gen,
            steps_per_epoch=math.floor(train_len / batch_size),
            epochs=params['max_epochs'],
            validation_data=val_gen,
            validation_steps=math.floor(val_len / batch_size),
            callbacks=[checkpointer, reduce_lr, stopping])

        util.save_train_loss(history, os.path.join(save_dir,  "losses.csv"), params['metrics'])

    finally:
        cleanup_old_checkpoints(save_dir, params['rename'])

def cleanup_old_checkpoints(model_dir, rename):  # TODO was doing thi
    " Delete epoch checkpoints in model save dir except for the last. Rename the run dir and the ckpt name to <rename> "
    print("Cleaning up checkpoints and renaming")
    checks = [o for o in os.listdir(model_dir) if o.endswith('hdf5')]
    # the way they saved autoencoder checkpoints is different from the way they saved siamese checkpoints...
    if len(checks[0].split('-')) == 3:  ix_epoch = 1
    else:  ix_epoch = 2
    epochs = [(i, int(e.split('-')[ix_epoch])) for i, e in enumerate(checks)]
    last_epoch = max(epochs, key=lambda x: x[1])
    for i, c in enumerate(checks):
        if i == last_epoch[0]:  
            old_ckpt_name = c
            continue
        os.remove(os.path.join(model_dir, c))
    # And now rename
    model_dir_basename = os.path.basename(model_dir)  # the numeric identifier for the run
    new_model_dir = model_dir.replace(model_dir_basename, rename)
    os.makedirs(new_model_dir, exist_ok=True)
    # now rename the run dir (well, everything in it) - 
    for f in os.listdir(model_dir):
        os.rename(os.path.join(model_dir, f), os.path.join(new_model_dir, f))
    # and now rename the ckpt too
    os.rename(os.path.join(new_model_dir, old_ckpt_name), os.path.join(new_model_dir, rename+".hdf5"))
    
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path to config file")
    parser.add_argument('--rename',  type=str, default='latest', help='If provided, renames the last epoch hdf5 to provided to make it easier to fine')
    args = parser.parse_args()
    params = json.load(open(args.config_file, 'r'))
    params['filename'] = str(args.config_file)
    params['rename'] = args.rename
    train(params)
