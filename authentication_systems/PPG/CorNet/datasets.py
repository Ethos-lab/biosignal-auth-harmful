import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

from custom_datasets import datasets, processing
import random

import neurokit2 as nk2

RESAMPLE_FS = 125 

def min_max_norm(sigs):
    new_sigs = []
    for s in sigs:
        mi, ma = s.min(), s.max()
        ns = (s-mi)/(ma-mi)
        new_sigs.append(ns)
    return new_sigs

class AllUsersDataset(Dataset):
    def __init__(self, dataset, split, fs=128, spoof_name='original', IGNORE_FRACTION_TIME=False):

        if split == 'train':
            fraction_time = (0.5, 0.625)
            fraction_time = (0.5, 0.75)
        elif split == 'test':
            fraction_time =  (0.625, 0.75)
            fraction_time = (0.75, 1.0)
        elif split == 'eval':
            fraction_time = (0.75, 1.0)
        else:
            raise ValueError
        if IGNORE_FRACTION_TIME:
            fraction_time = (0, 1.0) # just see if it can train 

        self.DATA, self.PATIENT_IX = [], []
        self.HRS = []

        orig_fs = datasets.get_orig_fs(dataset, datatype='ppg', spoof=spoof_name!='original', spoof_name=spoof_name)

        for pix, patient, ppg in datasets.get_data(dataset, 'ppg', spoof=spoof_name!='original', spoof_name=spoof_name, split='all', fraction_time=fraction_time): 

   
            ppg = processing.resample_signal(ppg, orig_fs, RESAMPLE_FS)  
            ppg = processing.zscore(ppg)  # NOTE moved monday from after filter to before. 
            ppg = processing.filter_signal(ppg, freqs=[0.1, 18], fs=orig_fs)

            segments = self.segment_ppg(ppg)  # defaults

            if dataset == 'dalia' or dataset == 'wesad' or dataset == 'ubfcphys' or dataset == 'pure':
                ppg = min_max_norm(ppg)  # Only ones that need it for the HR detection

            #segments = self.filter_segments(segments)  # NOTE addition for ubfcphys

            hrs = self.avg_hr_segment(segments)

            self.DATA.extend(segments)
            self.HRS.extend(hrs)
            self.PATIENT_IX.extend([pix for i in range(len(segments))])

        self.DATA = np.stack(self.DATA)
        self.DATA = np.expand_dims(self.DATA, 1)

    # Sement ppg into 'input dimensionality of 1000' at 125 Hz => 8sec 
    # windows end up being 8sec long (with 2sec overlap, or 250 sample overlap)
    def segment_ppg(self, ppg, width=1000, overlap=6.0, fs=RESAMPLE_FS): 
        segments = []
        i = 0
        lp = ppg.size
        shift_len = int(overlap*fs) # shift this much each time (should be 8sec-2sec = 6sec)
        while i+width < lp:
            j = i+width
            seg = ppg[0, i:j]
            segments.append(seg)
            i = j-shift_len
        return segments

    def avg_hr_segment(self, segments, fs=RESAMPLE_FS):
        hrs = []
        for s in segments:
            try:
                res = nk2.ppg_process(s, sampling_rate=fs)
                hr = res[0]['PPG_Rate'].mean()
            except Exception as e:
                # Too few peaks, make hr 0 so we dont get nan losses
                hr = 0

            if np.isnan(hr):  hr = 0  #same thing

            hrs.append(hr) 
        return hrs 

    def filter_segments(self, segments):
        sigs = [np.std(seg) for seg in segments]
        avg_sig = np.mean(sigs)
        remove = np.where([s > avg_sig*3 for s in sigs])
        import pdb; pdb.set_trace()
        return [seg for i, seg in enumerate(segments) if i not in remove]

    def __len__(self):
        return len(self.DATA)

    def __getitem__(self, ix):
        return self.DATA[ix], self.PATIENT_IX[ix], self.HRS[ix]


class SingleUserDataset(Dataset):
    def __init__(self, fulldset, user, expansion_len=100, baseline=False):
        """
        expansion_len here means how many passes through the data do we have for 1:1 self:other size
        expansion_len==-1 == expansion_len==2 cause of poor planning
        expansion_len==3 means that there's a 1:2 odds of getting self v getting someone else, and the size of the dset is
            adjusted for that 
        """
        self.dset = fulldset
        self.user = user
        self.BASELINE = baseline  # if baseline, then loads basically all the other users besides user

        # for convenience:
        all_ixs = list(range(len(self.dset)))
        if baseline:
            # Make it fair and only get the total num samples that spoof would have access to
            num_to_get = len([i for i, u in enumerate(self.dset.PATIENT_IX) if u == user])
            self.user_ixs = [i for i, u in enumerate(self.dset.PATIENT_IX) if u != user]
            random.shuffle(self.user_ixs)
            self.user_ixs = self.user_ixs[:num_to_get]
        else:
            self.user_ixs = [i for i, u in enumerate(self.dset.PATIENT_IX) if u == user]
        self.other_ixs = list(set(all_ixs) - set(self.user_ixs))  # other

        if expansion_len == -1:
            self.total_len = len(self.user_ixs)+len(self.other_ixs)  # one pass through the data
        else:        
            self.total_len = len(self.user_ixs)*expansion_len

    def __len__(self):
        if self.BASELINE:  return len(self.user_ixs)
        return self.total_len

    def __getitem__(self, ix):
        if self.BASELINE:  
            data, _, hr = self.dset[self.user_ixs[ix]] # these users are other
            return data, False, hr

        # Otherwise, get data with probability 
        if ix < len(self.user_ixs):
            is_self = True
            data, _, hr = self.dset[self.user_ixs[ix]]
        else:
            is_self = False
            # Let's try something new, March 7. Get a random one each time so that each epoch different
            rand_other = random.randint(0, len(self.other_ixs)-1)
            try:
                data, _, hr = self.dset[self.other_ixs[rand_other]]
            except:
                import pdb; pdb.set_trace()
            
            #ix = ix - len(self.user_ixs)  # to zero index from start of self.other_ixs
            #ix = ix % len(self.other_ixs)  # just in case we ask for more than we actually have
            #data, _, hr = self.dset[self.other_ixs[ix]]
        return data, is_self, hr

    

