"""
For each patient, train a binary classifier. 
This is actually what they do in the paper
"""

from datasets import MulticlassDataset, BinaryDataset

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from custom_datasets import datasets, processing
import os
import random
import models

import signal_processing as sp
from sklearn.metrics import roc_curve

from utils import train_epoch, test_epoch, eval_eer
from collections import defaultdict

from scipy.stats import binomtest


class BinaryDataset(BinaryDataset):
    def __init__(self, fulldataset, user, neg_multiplier=2.0, baseline=False, eval=True):
        super().__init__(fulldataset, user, neg_multiplier, baseline, eval=True)
        
    def create_examples(self, num_tries=50):
        # examples is a list of dicts with keys 'ix' and 'x'
        num_tries = min(len(self.user_ixs), num_tries)
        example_ixs = random.sample(self.user_ixs, k=num_tries)
        examples, already_tried = [], []
        for ix in example_ixs:
            item = {'ix': ix, 'x': self.dset[ix][0]}
            already_tried.append(ix)
            examples.append(item)
        return examples, already_tried

    def filter_examples(self, examples, y_pred, already_tried):
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
                    new_ix = random.sample(self.user_ixs, 1)[0]
                    tried_for_count += 1
                    if tried_for_count > 100: 
                        no_more_samples = True
                        #print("We ran out of samples; future trials re-using old ones")
                        break
                # Got a new one 
                data = self.dset[new_ix][0]
                new_examples.append({'ix': new_ix, 'x': data})
                already_tried.append(new_ix) 
        return new_examples, already_tried, no_more_samples

def predict(model, xs, eer_thresh, device):
    y_pred = []
    x = [ex['x'] for ex in xs]
    x = torch.Tensor(x).to(device)
    out = model(x)
    for o in out:
        if o >= eer_thresh:  # if round(1)=1 (or above thresh, pred me. else pred someone else
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred, np.mean(y_pred)
        

def eval(args):

    # "for the distacne matching approach, we get the average of the training data for a given user and use that as the template.
    # Somewhat annoying -- cant load dataset until we know the channels from laoding one of the models

    # Make sure that channels used to train on is correct
    saved_model = torch.load(os.path.join(args.models_dir, f'0.pt'))
    if 'channels' not in saved_model:
        channels = args.channels
    else:
        channels = saved_model['channels']

    spoof_dset = MulticlassDataset(args.dataset, 'eval', fs=128, spoof_name=args.spoof_name, channels=channels, augmentation=False)
    num_users = spoof_dset.num_users
    device = 'cuda:1'

    print("Evaluating models: ", args.models_dir)
    print("Spoof_name: ", args.spoof_name)

    ALL_TEST_ACCS = defaultdict(list)  # 
    ALL_Y_PRED = defaultdict(list)
    for user in set(spoof_dset.users):
        saved_model = torch.load(os.path.join(args.models_dir, f'{user}.pt'))
        channels = saved_model['channels']
        num_channels = saved_model['num_channels']
        lstm = saved_model['lstm']

        print("WARNING: KNOW THAT SAVED CHANNELS ARE: ", channels, " AND CHOSEN HERE WERE: ", spoof_dset.channels)

        user_spoof_dset = BinaryDataset(spoof_dset, user, neg_multiplier=1, baseline=args.baseline)  # only gen positive examples, from self or other user samples

        model = models.get_model(num_channels, 1, lstm=lstm, binary=True)
        model.to(device)
        model.load_state_dict(saved_model['model_state_dict'])
        model.eval()

        EER_THRESH = saved_model['EER_THRESH']

        print(f"\nFor user {user}: ", end='')
        examples, already_tried = user_spoof_dset.create_examples(args.num_matches)
        no_more_samples = False
        for i in range(args.num_trials):
            if i > 0:  # yeah i know this is awkward
                examples, already_tried, no_more_samples = user_spoof_dset.filter_examples(examples, y_pred, already_tried)
            #print('Picked: ', [e['ix'] for e in examples])
            y_pred, acc = predict(model, examples, EER_THRESH, device)
            #print('Ypred:  ', [y for y in y_pred])
            ALL_Y_PRED[i].extend(y_pred)

            print(f"{100*acc:.2f}", end=', ')
            #ALL_TEST_ACCS[i].append(acc)
            #print()

    print("\n\n")
    print("Average/std aross users: ")
    for i in range(args.num_trials):
        y_pred = ALL_Y_PRED[i]
        bt = binomtest(np.sum(y_pred), args.num_matches*len(set(spoof_dset.users)))
        ci = bt.proportion_ci(confidence_level=0.95)
        print(f"{np.mean(y_pred):.3f}\t{ci.low:.3f}\t{ci.high:.3f}")        

                    

if __name__ == '__main__':
    parser = ArgumentParser('Hwang 2020/2021 Wavelet PPG')
    parser.add_argument('--dataset', type=str, default='bidmc')
    parser.add_argument('--spoof_name', type=str, default='original')
    parser.add_argument('--models_dir', type=str, default='this dir should contain each user.pt')
    parser.add_argument('--num_trials', type=int, default=10)
    parser.add_argument('--num_matches', type=int, default=25, help='batch size for eval')
    parser.add_argument('--channels', type=str, nargs="+", default=['DTW', 'ZP'])
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()

    with torch.no_grad():
        eval(args)
