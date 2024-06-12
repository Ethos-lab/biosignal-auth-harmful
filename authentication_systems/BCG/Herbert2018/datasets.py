import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

from custom_datasets import datasets, processing
import random

import neurokit2 as nk2

def min_max_norm(segments):
    new_segments = []
    for s in segments:
        mi, ma = s.min(), s.max()
        news = (s - mi)/(ma - mi)
        new_segments.append(news)
    return new_segments

class AllUsersDataset(Dataset):
    def __init__(self, dataset, split, fs=128, spoof_name='original'):

        if split == 'train':
            fraction_time = (0.5, 0.625)
        elif split == 'test':
            fraction_time =  (0.625, 0.75)
        elif split == 'eval':
            fraction_time = (0.75, 1.0)
        else:
            raise ValueError

        self.DATA, self.PATIENT_IX = [], []
        self.HRS = []

        RESAMPLE_FS = 512
        #RESAMPLE_FS = 50  # what the apper did   # TODO changed Monday 

        spoof = spoof_name != 'original'
        orig_fs = datasets.get_orig_fs('bedbased', datatype='bcg', spoof=spoof, spoof_name=spoof_name)
        for pix, patient, bcg in datasets.get_data('bedbased', 'bcg', spoof=spoof, spoof_name=spoof_name, split='all', fraction_time=fraction_time):
        #for pix, patient, bcg in datasets.get_data('bedbased', 'bcg', spoof=spoof, spoof_name=spoof_name, fraction_time=fraction_time, subject_ix=['X0132', 'X1001', 'X1002']):

            bcg = processing.resample_signal(bcg, orig_fs, RESAMPLE_FS)  # TODO check fi 128 isA okay

            segments = self.segment_bcg(bcg, RESAMPLE_FS)  # 1) segmentation into 3sec with 2sec overlap at 50Hz
            segments = [processing.zscore(s) for s in segments]  2) 
            segments = self.rolling_average(segments, RESAMPLE_FS)
            segments = [processing.filter_signal(s, [4, 11], RESAMPLE_FS, 4) for s in segments] 
 
            self.DATA.extend(segments)
            self.PATIENT_IX.extend([pix for i in range(len(segments))])

            print(f"Processed data for patient: {patient}, size: {len(segments)}")

        self.DATA = np.stack(self.DATA)
        self.DATA = np.expand_dims(self.DATA, 1)
        self.SAMPLE_LEN = len(segments[0])  # for instantiaing the model

    def segment_bcg(self, bcg, fs=128):
        # Default 3 second segments with 2sec of overlap
        width = int(fs*3.0)
        shift = int(fs*1.0)
        segments = []
        i = 0
        lb = bcg.size
        while i+width < lb:
            j = i+width
            seg = bcg[0, i:j]
            segments.append(seg)
            i = j-shift
        return segments

    def __len__(self):
        return len(self.DATA)

    def __getitem__(self, ix):
        return self.DATA[ix], self.PATIENT_IX[ix]

    def rolling_average(self, segments, fs=128):
        # subtract a 35 sample mean . their orig fs was 50Hz, so this is 0.7sec
        window = int(0.2*fs)
        segs = []
        for s in segments:
            # pad repeat on either end 
            numpad = window - 1
            if numpad % 2 == 0:
                pads = np.pad(s, (int(numpad/2), int(numpad/2)), mode='edge')
            else:
                pads = np.pad(s, (int(numpad/2), int(numpad/2)+1), mode='edge')

            rollingavg = np.convolve(pads, np.ones(window), 'valid')/window
            ns = s - rollingavg
            segs.append(ns)
        return segs



class SingleUserDataset(Dataset):
    def __init__(self, fulldset, user, expansion_len=100, baseline=False):
        self.dset = fulldset
        self.user = user

        self.BASELINE = baseline

        # for convenience:
        all_ixs = list(range(len(self.dset)))
        if baseline:
            # first ge tthe num that we get with the other so it's fair
            num_to_sample = len([i for i, u in enumerate(self.dset.PATIENT_IX) if u == user])
            self.user_ixs = [i for i, u in enumerate(self.dset.PATIENT_IX) if u != user]
            random.shuffle(self.user_ixs)
            self.user_ixs = self.user_ixs[:num_to_sample]
        else:
            self.user_ixs = [i for i, u in enumerate(self.dset.PATIENT_IX) if u == user]
        self.other_ixs = list(set(all_ixs) - set(self.user_ixs))  # other

        self.total_len = len(self.user_ixs)*expansion_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, ix):
        """
        is_self = random.uniform(0,1) <= self.p
        if is_self:
            ix = ix % len(self.user_ixs)
            data, _ = self.dset[self.user_ixs[ix]]
        else:
            ix = ix % len(self.other_ixs)
            data, _ = self.dset[self.other_ixs[ix]]
        """

        # Honestly this is dumb
        if self.baseline:
            return self.user_ixs[ix]  # we should never have more than this
        else:

            if ix < len(self.user_ixs):
                data, _ = self.dset[self.user_ixs[ix]]
                is_self = True
            else:
                ix = ix % len(self.other_ixs)
                data, _ = self.dset[self.other_ixs[ix-len(self.user_ixs)]]
                is_self = False
                
            return data, is_self

    

