"""
For each patient, train a binary classifier. 
This is actually what they do in the paper
"""

from datasets import AllUsersDataset, SingleUserDataset

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
import os
import random
from model import Model

from sklearn.metrics import roc_curve

from utils import train_epoch, test_epoch, eval_eer
from collections import defaultdict

from scipy.stats import binomtest


def create_examples(dset, num_tries=10):
    # examples is a list of dicts with keys 'ix' and 'x'
    num_tries = min(len(dset), num_tries)
    example_ixs = random.sample(list(range(len(dset.user_ixs))), k=num_tries)
    examples, already_tried = [], []
    for ix in example_ixs:
        if ix > len(dset):  import pdb; pdb.set_trace()
        item = {'ix': ix, 'x': dset.dset[ix][0]}
        already_tried.append(ix)
        examples.append(item)
    return examples, already_tried

def filter_examples(dset, examples, y_pred, already_tried, no_more_samples=False):
    # Stared off with n samples. Get another batch for the ones that didn't work
    # examples will be a dict with ones
    new_examples = []
    no_more_samples = False
    for ie, ex in enumerate(examples):
        if y_pred[ie]:  # keep it
            new_examples.append(ex)
        else:
            # Replace it but also keep track of the other ones we've tried
            new_ix = ex['ix']
            tried_for_count = 0
            while new_ix in already_tried: # one we already tried should be in this
                new_ix = random.sample(list(range(len(dset))), 1)[0]  # BUG same
                tried_for_count += 1
                if tried_for_count > 100:
                    no_more_samples = True
                    #print("We ran out of samples; future trials re-using old ones")
                    break
            # Got a new one
            data = dset.dset[new_ix][0]
            new_examples.append({'ix': new_ix, 'x': data})
            already_tried.append(new_ix)
    return new_examples, already_tried, no_more_samples



def predict(model, xs, eer_thresh, device):
    y_pred = []
    x = [ex['x'] for ex in xs]
    x = torch.Tensor(x).to(device)
    _, out = model(x)
    for o in out:
        if o >= eer_thresh:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred, np.mean(y_pred)
        

def eval(args):

    # "for the distacne matching approach, we get the average of the training data for a given user and use that as the template."
    print("Running with num_matches (per patient): ", args.num_matches)
    spoof_dset = AllUsersDataset(args.dataset, 'eval', fs=128, spoof_name=args.spoof_name, IGNORE_FRACTION_TIME=args.IGNORE_FRACTION_TIME)
    device = 'cuda:0'  # eval on 0, train on 1

    print("Evaluating models: ", args.models_dir)
    print("Spoof_name: ", args.spoof_name)

    all_eers = []  # just for reporting again 

    ALL_Y_PRED = defaultdict(list)
    for user in set(spoof_dset.PATIENT_IX):
        saved_model = torch.load(os.path.join(args.models_dir, f'{user}.pt'))

        user_spoof_dset = SingleUserDataset(spoof_dset, user, expansion_len=1, baseline=args.baseline)  # only gen positive examples

        model = Model()
        model.to(device)
        model.load_state_dict(saved_model['model_state_dict'])
        model.eval()

        EER_THRESH = saved_model['EER_THRESH']
        all_eers.append(saved_model['EER'])

        examples, already_tried = create_examples(user_spoof_dset, args.num_matches)
        print(f"For user {user}: num_samples: {len(examples)}")
        no_more_samples = False
        for i in range(args.num_trials):
            if i > 0 and not no_more_samples:  # yeah i know this is awkward
                examples, already_tried, no_more_samples = filter_examples(user_spoof_dset, examples, y_pred, already_tried, no_more_samples)
            # otherwise try (again) with the examples we already have
            y_pred, acc = predict(model, examples, EER_THRESH, device)
            ALL_Y_PRED[i].extend(y_pred)


    print("\n\n")
    print("Average/std aross users: ")
    for i in range(args.num_trials):
        y_pred = ALL_Y_PRED[i]
        bt = binomtest(np.sum(y_pred), args.num_matches*len(set(spoof_dset.PATIENT_IX)))  # or just len(y_pred)
        ci = bt.proportion_ci(confidence_level=0.95)
        print(f"{np.mean(y_pred):.3f}\t{ci.low:.3f}\t{ci.high:.3f}")        


    print("Average train EER was: ", np.mean(all_eers)) 

if __name__ == '__main__':
    parser = ArgumentParser('Hwang 2020/2021 Wavelet PPG')
    parser.add_argument('--dataset', type=str, default='bidmc')
    parser.add_argument('--spoof_name', type=str, default='original')
    parser.add_argument('--models_dir', type=str, default='this dir should contain each user.pt')
    parser.add_argument('--num_trials', type=int, default=10)
    parser.add_argument('--num_matches', type=int, default=10, help='batch size for eval PER PATIENT')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--IGNORE_FRACTION_TIME', action='store_true')
    args = parser.parse_args()

    with torch.no_grad():
        eval(args)
