from argparse import ArgumentParser
import os
import numpy as np

import random
from scipy.stats import binomtest

from datasets import TrainDataset
import pickle

from collections import defaultdict

def predict(model, data, eer_thresh):
    scores = model.predict_proba(data)
    y_pred = [1 if y >= eer_thresh else 0 for y in scores[:,1]]
    return y_pred, scores

def subset_data(dset, u):
    ixs = [i for i, du in enumerate(dset.users) if du == u]
    subset = [dset.data[i] for i in ixs]
    return subset

def subset_other_data(dset, u):
    ixs = [i for i, du in enumerate(dset.users) if du != u]
    subset = [dset.data[i] for i in ixs]
    return subset

def eval(args):
    # Load the model and just predict on the eval_set. Don't need pairs. 
   
    eval_dset = TrainDataset(split='eval', spoof_name=args.spoof_name)
    ALL_Y_PRED = defaultdict(list)
    for u in set(eval_dset.users):

        with open(os.path.join(args.model_path, f'{u}.pkl'), 'rb') as fp:
            model_dict = pickle.load(fp)
            model = model_dict['model']
            EER_THRESH = model_dict['EER_THRESH']

        if args.baseline:
            subset = subset_other_data(eval_dset, u)
        else:
            subset = subset_data(eval_dset, u)

        all_sample_ixs = list(range(len(subset)))
        num_matches = 50  # TODO see how much data we have
        example_ixs = random.sample(all_sample_ixs, num_matches)
        for i in range(10):
            # Reduce all_sample_ixs (asmple with replacement from this)
            all_sample_ixs = [a for a in all_sample_ixs if a not in example_ixs]  # with replacement

            examples = [subset[e] for e in example_ixs]
            preds, scores = predict(model, examples, EER_THRESH)
            ALL_Y_PRED[i].extend(preds)

            # Filter examples
            for ei, a in enumerate(preds):
                if not a:  
                    if len(all_sample_ixs) == 0:  continue
                    example_ixs[ei] = random.sample(all_sample_ixs, 1)[0]

        
    for i in range(10):        
        y_pred = ALL_Y_PRED[i]
        bt = binomtest(np.sum(y_pred), num_matches*len(set(eval_dset.users)))
        ci = bt.proportion_ci(confidence_level=0.95)
        print(f"{np.mean(y_pred):.3f}\t{ci.low:.3f}\t{ci.high:.3f}")
            

    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--spoof_name', type=str, default='original')
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()

    # This goes by fast enough that we can just do both 
    save_spoof_name = args.spoof_name
    save_baseline = args.baseline
    args.spoof_name = 'original'
    args.baseline = False
    eval(args)
    args.spoof_name = save_spoof_name
    args.baseline = save_baseline
    eval(args)
