from torch.utils.data import Dataset
from signal_processing import SignalProcessor
import os
import glob
import pandas as pd
import numpy as np
from numpy import random
import itertools
from custom_datasets import datasets
from collections import Counter 
import neurokit2 as nk2

def zscore(sig):
    ' From cardiogan paper: perform person-specific zscore normalization. Also not what DeepECG does, but makes no sense without it '
    mean = sig.mean()
    std = sig.std()
    sig -= mean
    sig /= std
    return sig

class TrainDataset(Dataset):

    def __init__(self, name, split):
        # Dataset only for training the model, using train patients (DB_H_T)
        # Splits DB_H_T into train and test across time just so we can monitor training of the CNN
        # Has nothing to do with testing verification accuracy
        super().__init__()
        " See how far we get "


        patients = datasets.get_list_of_patients(name, spoof=False, split='all')  #split='train')
        fs = datasets.get_orig_fs(name, datatype='ecg', spoof=False)
        print("Fs: ", fs, " resampling to 128")
        self.num_train_classes = len(patients)

        # Length of each window to get a template for, in seconds. Paper used 10sec for 8 beats
        win_len = 15*fs  # NOTE changed from 10sec to 15sec since 8 beats was tight

        ECG_DATA, PATIENT_LABELS = [], []
        
        processor = SignalProcessor(fs=128, m=8, resample_from=fs)  # TODO NOTE always resampling to 128 from whatever it was

        if split == 'train':
            fraction_time = (0.5, 0.675) # this is 70% of the (0.5, 0.75) quarter
            #fraction_time = (0, 0.5)
        elif split == 'test':
            fraction_time = (0.675, 0.75) # this is 30% of the (0.5, 0.75) quarter
            #fraction_time = (0.5, 1.0)
        else:
            raise ValueError("Unknown split: " + split)

        print('Processed data for subject: ', end = '')
        for pix, patient, session, ecg in datasets.get_data(name, 'ecg', spoof=False, subject_ix=patients, fraction_time=fraction_time):

            ecg = zscore(ecg)
            ecg = ecg[0] # (num_leads, n) to (n,), first lead only
            if name == 'hcitagging':
                ecg = nk2.ecg_clean(ecg, sampling_rate=fs)

            #print(f"DEBUG: trace for pat {patient} is len: {ecg.size}, which at {fs}hz is {ecg.size/(fs*60)} min")

            # ADDED BECAUSE WE DO THIS FOR CARDIOGAN: ZSCORE PER PATIENT

            # Now segment into clips, each with the pix
            wix_start = 0
            win_len = 15*fs  # Reset 
            if win_len >= len(ecg):  win_len = len(ecg)-1  # Patch 
            while wix_start + win_len < len(ecg):
                seg = ecg[wix_start:wix_start+win_len]
                processed = processor.process(seg)  # this is where it gets resampled
                if processed is None:
                    print("Not enough peaks for pat: ", patient)
                    wix_start += win_len
                    continue

                ECG_DATA.append(processed)  # should be ~ 1x200
                PATIENT_LABELS.append(pix)

                wix_start += win_len
            print(patient, end=', ', flush=True)

        print()
    

        # Subtract each from the mean and multiply by 256:
        mean_ecg_data = np.mean(ECG_DATA, axis=0)
        NORMALIZED_ECG_DATA = [256*(e - mean_ecg_data) for e in ECG_DATA]

        self.ECG_DATA = np.array(NORMALIZED_ECG_DATA)
        self.ECG_DATA = np.expand_dims(self.ECG_DATA, 1)  # n x 1 x 200
        self.LABELS = np.array(PATIENT_LABELS)               
        try:
            self.sig_len = self.ECG_DATA.shape[2]
        except:
            import pdb; pdb.set_trace()


        print("Dataset loaded. Shape: ", self.ECG_DATA.shape)
        # Num per patient 
        num_per_patient = Counter(self.LABELS)
        for k, v in num_per_patient.items():
            print(k, ":", v)

    def __len__(self):
        return len(self.LABELS)

    def __getitem__(self, idx):
        return self.ECG_DATA[idx], self.LABELS[idx]

    def generate_pair(self, i=None, j=None):
        max_int = self.__len__() 
        if i is None:  i = random.randint(0, max_int)
        if j is None:  j = random.randint(0, max_int)
        return self.ECG_DATA[i], self.LABELS[i], self.ECG_DATA[j], self.LABELS[j]

    def generate_genuine_pair(self):
        all_pairs = []
        unique_patients = list(range(self.num_train_classes))
        for u in unique_patients:
            subset = [i for i, p in enumerate(self.LABELS) if p == u]
            pairs = itertools.product(subset, subset)
            all_pairs.extend(pairs)

        for p, j in all_pairs:
            if p == j:  continue # dont want the same pair, kind of cheating that way
            yield self.generate_pair(p, j)

    def generate_imposter_pair(self):
        unique_patients = list(range(self.num_train_classes))
        for u in unique_patients:
            subset = [i for i, p in enumerate(self.LABELS) if p != u]
            pairs = itertools.product(subset, subset)
            for p, j in pairs:
                yield self.generate_pair(p, j)

class EvalDataset(Dataset):
    # Is a Dataset, but needs to generate genuine and imposter pairs of x data, no y data really
    # Used to get EER threshold

    def __init__(self, name, spoof=False, spoof_name=None, split='all'):
        super().__init__()

        if split == 'test':
            fraction_time = (0.75, 1.0)
        elif split == 'all':
            raise NotImplementedError  # deprecated error actually
        else:  # default here
            fraction_time = (0.75, 1.0)
        

        # Use the entire trace for test data, not need to split in time
        patients = datasets.get_list_of_patients(name, spoof=spoof, spoof_name=spoof_name, split='all')
        self.num_test_classes = len(patients)

        fs = datasets.get_orig_fs(name, datatype='ecg', spoof=spoof, spoof_name=spoof_name)

        win_len = 15*fs  # This happens before the processor
        print("fs: ", fs, " being resampled to 128")

        RAW_DATA = []
        ECG_DATA, PATIENT_LABELS = [], []

        processor = SignalProcessor(fs=128, m=8, resample_from=fs)
        for pix, patient, session, ecg in datasets.get_data(name, 'ecg', spoof, spoof_name, split='all', fraction_time=fraction_time):

            # ADDED BECAUSE WE DO THIS FOR CARDIOGAN: ZSCORE PER PATIENT
            ecg = zscore(ecg)
            ecg = ecg[0]
            #print(f"DEBUG: trace for pat {patient} is len: {ecg.size}, which at {fs}hz is {ecg.size/(fs*60)} min")
            if name == 'hcitagging':
                ecg = nk2.ecg_clean(ecg, sampling_rate=fs)

            wix_start, wix_end = 0, len(ecg)
            win_len = 15*fs  # Reset 
            if win_len >= len(ecg):  win_len = len(ecg)-1  # Patch 
            while wix_start + win_len < wix_end:
                seg = ecg[wix_start:wix_start+win_len]
                processed = processor.process(seg)

                RAW_DATA.append(seg)
                ECG_DATA.append(processed)
                PATIENT_LABELS.append(pix)

                wix_start += win_len

                if processed is None:  print("Sig too short for patient ", patient)

        mean_ecg_data = np.mean(ECG_DATA, axis=0)
        NORMALIZED_ECG_DATA = [256*(e - mean_ecg_data) for e in ECG_DATA]
        self.ECG_DATA = np.array(NORMALIZED_ECG_DATA)
        self.ECG_DATA = np.expand_dims(self.ECG_DATA, 1)  # n x 1 x 200
        self.LABELS = np.array(PATIENT_LABELS)
        self.RAW_DATA = RAW_DATA # just for debugging
        self.sig_len = self.ECG_DATA.shape[2]

        print("Dataset loaded. Shape: ", self.ECG_DATA.shape)
        num_per_patient = Counter(self.LABELS)
        for k, v in num_per_patient.items():
            print(k, ":", v)


    def __len__(self):
        return len(self.LABELS)

    def __getitem__(self, idx):
        return self.ECG_DATA[idx], self.LABELS[idx]

    def generate_pair(self, i=None, j=None):
        max_int = self.__len__()
        if i is None:  i = random.randint(0, max_int)
        if j is None:  j = random.randint(0, max_int)
        return self.ECG_DATA[i], self.LABELS[i], self.ECG_DATA[j], self.LABELS[j]

    def generate_genuine_pair(self):
        all_pairs = []
        unique_patients = list(range(self.num_test_classes))
        for u in unique_patients:
            subset = [i for i, p in enumerate(self.LABELS) if p == u]
            pairs = itertools.product(subset, subset)
            all_pairs.extend(pairs)
        random.shuffle(all_pairs)

        for p, j in all_pairs:
            if p == j:  continue
            yield self.generate_pair(p, j)

    def generate_imposter_pair(self):
        all_pairs = []
        unique_patients = list(range(self.num_test_classes))
        for u in unique_patients:
            subset = [i for i, p in enumerate(self.LABELS) if p != u]
            pairs = itertools.product(subset, subset)
            all_pairs.extend(pairs)
        random.shuffle(all_pairs)

        for p, j in all_pairs:
            yield self.generate_pair(p, j)



class DualEvalDataset:
    # Same thing as Eval Dataset but we can load 2 different datasets for comparison. 
    # Also only gives us (attempted) positive pairs, because that's what we're interested in

    def __init__(self, name,  spoof_name1, spoof_name2, baseline=False):
        super().__init__()
        self.dataset1 = EvalDataset(name, spoof=spoof_name1 != 'original', spoof_name=spoof_name1, split='eval')
        self.dataset2 = EvalDataset(name, spoof=spoof_name2 != 'original', spoof_name=spoof_name2, split='eval')

        # Set and save all pairs here
        all_pairs = []
        users1 = self.dataset1.LABELS
        users2 = self.dataset2.LABELS
        unique_patients = set(users1)

        if not baseline:
            for u in unique_patients:
                subset1 = [i for i, p in enumerate(users1) if p == u]
                subset2 = [i for i, p in enumerate(users2) if p == u]
                pairs = itertools.product(subset1, subset2)
                all_pairs.extend(pairs)
        else:
            for u in unique_patients:
                subset1 = [i for i, p in enumerate(users1) if p == u]
                subset2 = [i for i, p in enumerate(users2) if p != u]
                # Clip because we'll run out of memory or something
                random.shuffle(subset2)
                subset2 = subset2[:len(subset1)]
                pairs = itertools.product(subset1, subset2)
                all_pairs.extend(pairs)

        random.shuffle(all_pairs)

        self.users1 = users1
        self.users2 = users2
        self.all_pairs = all_pairs
    
    def generate_genuine_pair(self):
        for i1, i2 in self.all_pairs:
            yield self.complete_pair(i1, i2) 

    def get_another_partner(self, ix1, ix2):
        # Generates another pair where the first ex stays the same but the second ix changes
        raise NotImplementedError

    def get_another_pair(self, ix1):
        subset = [a for a in self.all_pairs if a[0] == ix1]
        another_match = random.randint(0, len(subset)-1)
        return self.complete_pair(ix1, subset[another_match][1])

    def complete_pair(self, i1, i2):
        user1 = self.users1[i1]
        user2 = self.users2[i2]
        sig1 = self.dataset1.ECG_DATA[i1]
        sig2 = self.dataset2.ECG_DATA[i2]
        return {'ix1': i1,
                'ix2': i2,
                'user': user1,
                'sig1': sig1,
                'sig2': sig2 }
    



