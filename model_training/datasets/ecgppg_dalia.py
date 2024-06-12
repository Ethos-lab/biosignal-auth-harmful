"""
Define Dataset class -- returns a dictionary with a single ECG and a single PPG
"""

import glob
import os
import numpy as np
import pickle as pkl

import pandas as pd

from datasets.dataset_utils import sliding_window, resample
from datasets.dataset_utils import filter_ecg, filter_ppg
from datasets.dataset_utils import zscore, min_max_norm

from datasets.base_dual_dataset import BaseDualDataset
from datasets.dataset_metadata import DualDatasetMetadata

from custom_datasets import datasets

class DualDataset(BaseDualDataset):

    METADATA = DualDatasetMetadata(
        NAME_A = 'ECG',
        NAME_B = 'PPG',
        NUM_LEADS_A = 1,
        NUM_LEADS_B = 1,
        FS_A = 128,
        FS_B = 128,
        SAMPLE_LEN = 512,
        NUM_SAMPLES_TRAIN = 14386, 
        NUM_SAMPLES_TEST = 3620
    )



    def __init__(self, split="train", evaluate=False, fraction_time=True):
        super().__init__(split, evaluate, fraction_time)

    def get_gt_hrs(self, patient):
        print('Loading GT for: ', patient)
        base_path = datasets.get_base_path('dalia')
        s = os.path.join(base_path, patient, patient+".pkl")
        with open(s, 'rb') as fp:
            data = pkl.load(fp, encoding='latin1')
        seconds = data['rpeaks']/700  # rpeaks is ix of the gt peaks, from ecg
        diffs = np.diff(seconds)
        seconds = [sec for i, sec in enumerate(seconds[1:]) if diffs[i] > 1e-6]  # mask where diffs nonzero
        diffs = [d for d in diffs if d > 1e-6]
        bpm = [60 / d for d in  diffs]
        df = pd.DataFrame({'s': seconds, 'bpm': bpm})
        return df

    def load_from_files(self):
        """
        sourced from here: https://archive.ics.uci.edu/ml/datasets/PPG-DaLiA

        There are 15 subjects - data for each subject in a different folder
        Collected during various activites -- currently will just aggregate all 
    
        Two devices: the RespiBAN chest ecg and the Empatica E4 wrist ppg
        RespiBAN has ECG, respiration, and accelerometer. 3-point ECG, 700Hz. 
        Empatica has PPG, accelerometer, EDA, TEMP. the ppg channel is called BVP at 64 Hz
        """


        ECG, PPG = [], []
        LABELS = []
        PATIENT_NAMES = []
        ECG_FS = datasets.get_orig_fs('dalia', 'ecg') # 700
        PPG_FS = datasets.get_orig_fs('dalia', 'ppg')  # 64
        TARGET_FS = self.METADATA.FS_A  # 128
        overlap_len = 0.4 if not self.evaluate else 0.0

        for pix, p, sess, data in datasets.get_paired_data('dalia', ['ecg', 'ppg'], spoof=False, split=self.split, fraction_time=(0,0.5)):

            ecg = data['ecg']
            ppg = data['ppg']

            # Following steps outlined in Cardiogan paper, section 4.2
            ecg = resample(ecg, orig_fs=ECG_FS, target_fs=TARGET_FS)
            ecg = filter_ecg(ecg, TARGET_FS)
            ecg = zscore(ecg)
            ecg = sliding_window(ecg, fs=TARGET_FS, window_len=4, overlap_len=overlap_len)
            ecg = min_max_norm(ecg)

            ppg = resample(ppg, orig_fs=PPG_FS, target_fs=TARGET_FS)
            ppg = filter_ppg(ppg, TARGET_FS)
            ppg = zscore(ppg)
            ppg = sliding_window(ppg, fs=TARGET_FS, window_len=4, overlap_len=overlap_len)
            ppg = min_max_norm(ppg)


            print(f" which generated {ecg.shape[0]} samples")

            ECG.append(ecg)
            PPG.append(ppg)
            LABELS.append([pix for i in range(len(ecg))])
            PATIENT_NAMES.append(p)

        self.A = np.concatenate(ECG)
        self.B = np.concatenate(PPG)
        self.LABELS = np.concatenate(LABELS)
        self.PATIENT_NAMES = PATIENT_NAMES
        print("Loaded data of size: ", self.A.shape)
