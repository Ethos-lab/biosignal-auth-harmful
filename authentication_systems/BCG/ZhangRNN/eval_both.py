from dataset_both import TrainDataset  # basically can spoof both ECG and BCG from PPG
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from argparse import ArgumentParser
import os
import numpy as np

from model import Model
from custom_datasets import datasets, processing

from utils import train_epoch, test_epoch, eval_eer
import random
from collections import defaultdict

from scipy.stats import binomtest

class PairGenerator:
    def __init__(self, dset1, dset2, baseline=False):
        self.dset1 = dset1
        self.dset2 = dset2

        self.l1 = len(dset1)
        self.l2 = len(dset2)

        self.baseline = baseline

        # Just print out stats:
        num_possible_pairs = 0
        for u in set(self.dset1.PATIENT_IX):
            p1 = sum([1 for p in self.dset1.PATIENT_IX if p == u])
            if baseline:
                p2 = sum([1 for p in self.dset2.PATIENT_IX if p != u])
            else:
                p2 = sum([1 for p in self.dset2.PATIENT_IX if p == u])
            num_possible_pairs += (p1*p2)
        print("Num possible positive pairs: ", num_possible_pairs)

    def create_examples(self, num_matches=1000):
        # Sample <num_matches> orig samples and match with one corresponding spoofed to start
        # This is called to initialize the examples, called just once
        select_ix1 = random.sample(range(self.l1), k=num_matches)

        examples = []
        for ix1 in select_ix1:
            try:
                pat1 = self.dset1.PATIENT_IX[ix1]
                if self.baseline:
                    subset2 = [i for i, pat2 in enumerate(self.dset2.PATIENT_IX) if pat1 != pat2]
                else:
                    subset2 = [i for i, pat2 in enumerate(self.dset2.PATIENT_IX) if pat1 == pat2]
                ix2 = random.choice(subset2)
                ex = {'ix1': ix1, 'data1': self.dset1.DATA[ix1], 'ix2': ix2, 'data2': self.dset2.DATA[ix2], 'old_ix2': []}
            except Exception:
                import pdb; pdb.set_trace()
            examples.append(ex)

        return examples  # remember this happens in-place anyway so dont really need to return it

    def filter_examples(self, examples, y_pred):
        for ie, ex in enumerate(examples):
            if y_pred[ie]:  # keep it
                continue
            else:
                # Get another ix2 (that we haven't tried before) for this ix1
                ix1, ix2 = ex['ix1'], ex['ix2']
                ex['old_ix2'].append(ix2)
                pat1 = self.dset1.PATIENT_IX[ix1]
                if self.baseline:
                    subset2 = [i for i, pat2 in enumerate(self.dset2.PATIENT_IX) if pat1 != pat2]
                else:
                    subset2 = [i for i, pat2 in enumerate(self.dset2.PATIENT_IX) if pat1 == pat2]
                num_attempts = 0  # just in case we dont have enough data and the while loop goes forever
                #import pdb; pdb.set_trace()
                while ix2 in ex['old_ix2']:
                    num_attempts += 1
                    ix2 = random.choice(subset2)
                    if num_attempts > 10:
                        ix2 = ex['old_ix2'][-1]  # give up on it
                        break
                ex['ix2'] = ix2
                ex['data2'] = self.dset2.DATA[ix2]
                examples[ie] = ex  # just to be sure? idk
        return examples



def predict(model, eer_thresh, examples, device):
    scores = []
    with torch.no_grad():
        for i, e in enumerate(examples):
            x1 = torch.FloatTensor(e['data1']).to(device).unsqueeze(0)
            x2 = torch.FloatTensor(e['data2']).to(device).unsqueeze(0)
            feat1 = model.get_template(x1)
            feat2 = model.get_template(x2)
            rmse = np.linalg.norm(feat2.cpu()-feat1.cpu(), 2, axis=1)
            rmse = np.clip(rmse, 1e-5, rmse.max())
            rmse = 1/rmse
            scores.append(rmse.tolist())
        y_pred = [1 if s > eer_thresh else 0 for s in scores]
    return y_pred

def eval(args):
   
    model_dict = torch.load(args.model_path)
    T = model_dict['T']
    num_classes = model_dict['num_classes'] 
    eer = model_dict['EER']
    eer_thresh = model_dict['EER_THRESH']

    print(f"Loaded saved model that had T: {T}, eer: {eer}, eer_thresh: {eer_thresh}")

    orig_dset  = TrainDataset(split='eval', T=T, spoof_name='original')
    spoof_dset = TrainDataset(spoof_name=args.spoof_name, split='eval', T=T, both=args.both, ecg_spoof_name=args.ecg_spoof_name)

    device = 'cuda:1'

    model = Model(T=T, num_classes=num_classes, input_size=model_dict['input_size'])
    model.load_state_dict(model_dict['model_state_dict'])
    model.to(device)
    model.eval()

    num_repeats = 1
    num_attempts = 10
    num_matches = 500

    pair_gen = PairGenerator(orig_dset, spoof_dset, args.baseline)

    examples = pair_gen.create_examples(num_matches)
    for na in range(num_attempts):
        y_pred = predict(model, eer_thresh, examples, device)

        bt = binomtest(np.sum(y_pred), num_matches)
        ci = bt.proportion_ci(confidence_level=0.95)
        print(f"{np.mean(y_pred):.3f}\t{ci.low:.3f}\t{ci.high:.3f}")

        examples = pair_gen.filter_examples(examples, y_pred)

    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--spoof_name', type=str, default='original')
    parser.add_argument('--source_type', type=str, default='ecg')
    parser.add_argument('--both', action='store_true')
    parser.add_argument('--ecg_spoof_name', type=str, default='original')
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    assert not (args.both and args.ecg_spoof_name == 'original')
    eval(args)  # hope this works
