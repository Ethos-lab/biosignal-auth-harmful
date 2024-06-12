import argparse
import util
import loader
import glob
import os
from keras import backend as K
from custom_eer_metric import EER
import collections
import random
import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import binomtest
from tensorflow.keras import models, metrics
import json

import random

def predict(model, test_dset, eer_thresh):
    y_pred = []
    inputs = np.stack([b['match_sample'] for b in test_dset])
    out = model.predict(inputs, verbose=0)
    for y in out[:,1]: # how their predict.py did it
        if y >= eer_thresh: 
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred

def filter_examples(examples, y_pred, example_bank, first_create=False):
    if first_create:
        num_matches = len(y_pred)
        examples = [example_bank.pop(0) for e in range(num_matches)]
    else:
        for yi, y in enumerate(y_pred):
            if y:  continue  # keep that example
            elif len(example_bank) == 0:  continue # no more
            else:
                old_ex = examples[yi]
                new_ex = example_bank.pop(0)  # next one
                examples[yi] = new_ex
    return examples, example_bank  # i mean it's all in-lace so not really needed but okay

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--model_name', help='Will load saved_<dataset>/verification_<dataset>/<model_name>/<model_name>.hdf5')
    parser.add_argument('--spoof_name', default='original', help='Loads the second comparison dataset according to configs/siamese/<dataset>/config_test_<spoof_name>.json')
    parser.add_argument('--num_matches', default=100)
    parser.add_argument('--baseline', action='store_true')
    

    args = parser.parse_args()
    dataset = args.dataset

    model_path = os.path.join(f'saved_{dataset}', f'verification_{dataset}', args.model_name, f'{args.model_name}.hdf5')
    print("LOADING: ", model_path)
    preproc = util.load(os.path.dirname(model_path))
    model = models.load_model(model_path, custom_objects={'K': K, 'EER': EER})    

    # load eer and thresh
    model_path_dir = os.path.dirname(model_path)
    eer = np.load(os.path.join(model_path_dir, 'EER.npy'))
    eer_thresh = np.load(os.path.join(model_path_dir, 'EER_THRESH.npy'))


    orig_test_config = os.path.join('configs', 'siamese', dataset, f'config_test_original.json') # debugging
    spoof_test_config = os.path.join('configs', 'siamese', dataset, f'config_test_{args.spoof_name}.json')
    params1 = json.load(open(orig_test_config, 'r'))
    params2 = json.load(open(spoof_test_config, 'r'))

    # Load test set?
    test_dset = loader.prepare_dataset(params1['test'], params1)  # orig datasret
    spoof_dset = loader.prepare_dataset(params2['test'], params2)  # spoof (or whatever) dataset

    # Generate only positive pairs
    
    # it's num_positive PER USER christ
    example_bank = util.custom_create_positive_sample_dualdataset(preproc, test_dset, spoof_dset, 100, baseline=args.baseline)
    random.shuffle(example_bank)
    print("Total len of example bank: ", len(example_bank))

    examples, y_pred = [None]*args.num_matches, [None]*args.num_matches
    for i in range(10):
        examples, example_bank = filter_examples(examples, y_pred, example_bank, first_create = i == 0)
        y_pred = predict(model, examples, eer_thresh)
        bt = binomtest(np.sum(y_pred), len(y_pred))
        ci = bt.proportion_ci(confidence_level=0.95)
        print(f"{np.mean(y_pred):.3f}\t{ci.low:.3f}\t{ci.high:.3f}")
        

    print(f"EER was: {eer:.5f}, EER_THRESH was: {eer_thresh:.5f}")

