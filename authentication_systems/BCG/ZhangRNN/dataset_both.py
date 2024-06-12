from torch.utils.data import Dataset, DataLoader

import os
import numpy as np

from custom_datasets import datasets, processing

import random

class TrainDataset(Dataset):
    def __init__(self, split, both=False, T=15, spoof_name='original', ecg_spoof_name='original'):
        # Both means that both ECG and BCG are spoofed from PPG
        # Not both means that just BCG is spoofed from spoof_name

        self.T = T

        if split == 'train':
            fraction_time = (0.5, 0.625)
        elif split == 'test':
            fraction_time =  (0.625, 0.75)
        elif split == 'eval':
            fraction_time = (0.75, 1.0)
        else:
            raise ValueError

        self.DATA, self.PATIENT_IX = [], []

        spoof_ecg = both
        spoof = spoof_name != 'original'
        fs_ecg = datasets.get_orig_fs('bedbased', datatype='ecg', spoof=spoof_ecg, spoof_name=ecg_spoof_name)  # if not spoof_ecg, then spoof_name is ignored

        fs_bcg = datasets.get_orig_fs('bedbased', datatype='bcg', spoof=spoof, spoof_name=spoof_name)  # may be true or not

        patients = datasets.get_list_of_patients('bedbased', split='all', spoof=False)  # spoof doesn't matter here
        for pix, patient in enumerate(patients):
            ecggen = datasets.get_data('bedbased', 'ecg', spoof=spoof_ecg, spoof_name=ecg_spoof_name, subject_ix=[patient], fraction_time=fraction_time)
            _, _, ecg = next(ecggen)

            bcggen = datasets.get_data('bedbased', 'bcg', spoof=spoof_ecg, spoof_name=spoof_name, subject_ix=[patient], fraction_time=fraction_time)
            _, _, bcg = next(bcggen)

            bcg = processing.filter_signal(bcg, freqs=[1.0, 30.0], fs=fs_bcg)
            bcg = processing.zscore(bcg)

            bcg = processing.resample_signal(bcg, fs_bcg, 500)
            ecg = processing.resample_signal(ecg, fs_ecg, 500)
            rlocs = processing.get_ecg_rpeaks_hamilton(ecg, 500)

            segmented = self.segment_bcg(bcg, rlocs)
            segmented_ecg = self.segment_ecg(ecg, rlocs)

            group_segments = self.group_segments_both(segmented, segmented_ecg)


            #if len(group_segments) == 0:  import pdb; pdb.set_trace()
            self.DATA.extend(group_segments)
            self.PATIENT_IX.extend([pix for i in range(len(group_segments))])

            self.SAMPLE_LEN = group_segments[0].shape[1]

            print(f"Processed data for patient: {patient}, num heartbeats: {len(segmented)}")

    def group_segments_both(self, segmentsb, segmentse):
        group_segs = []
        for si in range(len(segmentsb)):
            if si + self.T >= len(segmentsb):  break
            seg = np.concatenate([segmentsb[si:si+self.T], segmentse[si:si+self.T]], axis=2).squeeze(1)
            group_segs.append(seg)
        return group_segs

    def __len__(self):
        return len(self.DATA)

    def __getitem__(self, ix):
        # Get a sequence of T heartbeats from the same person
        return self.DATA[ix], self.PATIENT_IX[ix]

        return np.concatenate(data, 0), user

    def segment_bcg(self, bcg, rlocs):

        heartbeats = []
        l = bcg.size
        for r in rlocs:
            if r+450 > l:  break
            hb = bcg[:, r:r+450]
            heartbeats.append(hb)

        return heartbeats

    def segment_ecg(self, ecg, rlocs):
        heartbeats = []
        l = ecg.size
        for r in rlocs:
            if r-150 < 0 or r+200 > l:  break
            hb = ecg[:, r-150:r+300]
            heartbeats.append(hb)
        return heartbeats

