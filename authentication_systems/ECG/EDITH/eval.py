import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
import math
from custom_datasets import datasets, processing
from torchmodels import BaseModel, SiameseModel, init_weights
from segment_signals import segmentSignals
from train_utils import train_siamese_epoch, test_siamese_epoch, eval_eer
import random
from collections import defaultdict
import itertools

from scipy.stats import binomtest
import neurokit2 as nk2

class SiameseDataset(Dataset):

    def __init__(self, name, split, base_model, negative_prob=0.5, spoof=False, spoof_name='original'):
        super().__init__()

        self.NEGATIVE_PROB = negative_prob

        # NOTE important, using all patients but (0.5, 0.625) for train, (0.625, 0.75) for crossval
        if split == 'train':
            fraction_time = (0.5, 0.675)
        elif split == 'test':
            fraction_time = (0.675, 0.75)
        elif split == 'eval':
            fraction_time = (0.75, 1.0)
        
        self.DATA, self.PATIENT_IX = [], []

        spoof = spoof if spoof_name != 'original' else False
        fs = datasets.get_orig_fs(name, datatype='ecg', spoof=spoof, spoof_name=spoof_name)  # get orig fs for resampling to 128
        for pix, patient, sess, ecg in datasets.get_data(name, 'ecg', spoof=spoof, spoof_name=spoof_name, split='all', fraction_time=fraction_time):
            #ecg, r_peaks = processing.resample_get_rpeaks(name, patient, ecg, resample_to=128)  # needs an array of r_peaks
            if name == 'hcitagging':
                ecg = nk2.ecg_clean(ecg[0], sampling_rate=fs)
                ecg = np.expand_dims(ecg, 0)

            ecg, r_peaks = processing.resample_get_rpeaks(ecg, fs, resample_to=128)
            segments, r_peaks = segmentSignals(ecg, r_peaks, person_id=pix, FS=128)
            segments = np.expand_dims(segments, 1)  # num x 1 x 256
            self.DATA.extend(segments)
            self.PATIENT_IX.extend([pix for i in range(len(segments))])
        self.DATA = np.stack(self.DATA)
        self.NUM_PATIENTS = len(set(self.PATIENT_IX))
        self.PATIENT_IX = np.array(self.PATIENT_IX)

        self.DATA = self.get_features(base_model, self.DATA)
        print('Loaded ' + split + ' data of size: ' + str(self.DATA.shape))

    def get_features(self, model, data):
        device = next(model.parameters()).device
        all_out = []
        batch_start, batch_end, batch_size = 0, 256, 256
        with torch.no_grad():
            while batch_end < len(data):
                batch = torch.FloatTensor(data[batch_start:batch_end,:,:])
                batch = torch.FloatTensor(batch).to(device)
                out = model.get_embedding(batch)
                all_out.append(out.cpu().numpy())
                batch_start += batch_size
                batch_end += batch_size
            if batch_start < len(data):
                batch = torch.FloatTensor(data[batch_start:,:,:])
                batch = torch.FloatTensor(batch).to(device)
                out = model.get_embedding(batch)
                all_out.append(out.cpu().numpy())
        all_out = np.concatenate(all_out)
        return all_out

    def __len__(self):
        return len(self.DATA)

    def __getitem__(self, ix):
        return self.DATA[ix]

class PairGenerator:
    def __init__(self, dset1, dset2, baseline=False, baseline_pat=None):
        self.dset1 = dset1
        self.dset2 = dset2

        self.l1 = len(dset1)
        self.l2 = len(dset2)

        self.BASELINE = baseline
        self.BASELINE_PAT = baseline_pat

        # Just print out stats:
        num_possible_pairs = 0
        for u in set(self.dset1.PATIENT_IX):
            p1 = sum([1 for p in self.dset1.PATIENT_IX if p == u])
            if baseline:
                if baseline_pat:
                    p2 = sum([1 for p in self.dset2.PATIENT_IX if p == baseline_pat])
                else:
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
            pat1 = self.dset1.PATIENT_IX[ix1]
            if self.BASELINE:
                if self.BASELINE_PAT:
                    subset2 = [i for i, pat2 in enumerate(self.dset2.PATIENT_IX) if pat2 == self.BASELINE_PAT]
                else:
                    subset2 = [i for i, pat2 in enumerate(self.dset2.PATIENT_IX) if pat1 != pat2]
            else:
                subset2 = [i for i, pat2 in enumerate(self.dset2.PATIENT_IX) if pat1 == pat2]
            ix2 = random.choice(subset2)
            ex = {'ix1': ix1, 'data1': self.dset1.DATA[ix1], 'ix2': ix2, 'data2': self.dset2.DATA[ix2], 'old_ix2': []}
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
                if args.baseline:
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
    # examples is a list [x1, x2] where each is jsut a single ndarray
    scores = []
    for i, e in enumerate(examples):
        x1 = torch.FloatTensor(e['data1']).to(device)
        x2 = torch.FloatTensor(e['data2']).to(device)
        out = model(x1.unsqueeze(0), x2.unsqueeze(0))
        scores.append(out.cpu().item())
    y_pred = [1 if s > eer_thresh else 0 for s in scores]
    return y_pred

def run_attempts(model, eer_thresh, device, eval_dset, spoof_dset, num_matches, baseline, base_pat):
    results = []
    num_attempts = 10 
    num_matches = num_matches
    
    if base_pat:  print("===== For base_pat: ", base_pat)
    pair_gen = PairGenerator(eval_dset, spoof_dset, baseline=baseline, baseline_pat=base_pat)
    examples = pair_gen.create_examples(num_matches)
    for nr in range(num_attempts):
        y_pred = predict(model, eer_thresh, examples, device)
        bt = binomtest(np.sum(y_pred), num_matches)
        ci = bt.proportion_ci(confidence_level=0.95)

        print(f"{np.mean(y_pred):.3f}\t{ci.low:.3f}\t{ci.high:.3f}")
        results.append((np.mean(y_pred), ci.low, ci.high))

        examples = pair_gen.filter_examples(examples, y_pred)
    return results 


def eval_siamese(dataset, base_model_fn, siamese_model_fn, device='cuda:0', spoof_name='original', num_matches=100, num_repeats=100, baseline=False, baseline_per_pat=False):

    base_model_dict = torch.load(base_model_fn)
    seq_len, n_classes = base_model_dict['seq_len'], base_model_dict['num_train_classes']
    base_model = BaseModel(seq_len=seq_len, n_classes=n_classes)
    base_model.load_state_dict(base_model_dict['model_state_dict'])
    base_model.to(device)
    base_model.eval()



    eval_dset = SiameseDataset(dataset, 'eval', base_model)
    spoof_dset = SiameseDataset(dataset, 'eval', base_model, spoof=True, spoof_name=spoof_name)
    print("Num patients: ", eval_dset.NUM_PATIENTS, spoof_dset.NUM_PATIENTS)

    # Load saved base classifier
    siamese_model_dict = torch.load(siamese_model_fn)
    model = SiameseModel()
    model.load_state_dict(siamese_model_dict['model_state_dict'])
    eer = siamese_model_dict['EER']
    print("Loaded model with EER: ", eer)
    eer_thresh = siamese_model_dict['EER_THRESH']
    model.to(device)
    model.eval()

    means, ci_lows, ci_highs = defaultdict(list), defaultdict(list), defaultdict(list) # attempt -> list of means, lows, highs
    if baseline_per_pat:
        # FOr each pat:
        for base_pat in set(eval_dset.PATIENT_IX):
            results = run_attempts(model, eer_thresh, device, eval_dset, spoof_dset, num_matches, baseline, base_pat)
            for ri, r in enumerate(results):
                means[ri].append(r[0])
                ci_lows[ri].append(r[1])
                ci_highs[ri].append(r[2])
       
        print("="*20) 
        for i in range(10): # just hardcoding this now 
            print(f"{np.mean(means[i]):.3f}\t{np.mean(ci_lows[i]):.3f}\t{np.mean(ci_highs[i]):.3f}")
    else:
        run_attempts(model, eer_thresh, device, eval_dset, spoof_dset, num_matches, baseline, None)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bidmc')
    parser.add_argument('--saved_base_model', type=str, default='')
    parser.add_argument('--saved_siamese_model', type=str, default='')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--spoof_name', type=str, default='original')
    parser.add_argument('--num_matches', type=int, default=500)
    parser.add_argument('--num_repeats', type=int, default=1)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--baseline_per_pat', action='store_true')
    args = parser.parse_args() 
    device = 'cpu' if args.cpu else 'cuda:0'
    print("Running eval with num_matches: ", args.num_matches)
    eval_siamese(args.dataset, args.saved_base_model, args.saved_siamese_model, device, args.spoof_name, args.num_matches, args.num_repeats, args.baseline, args.baseline_per_pat)
