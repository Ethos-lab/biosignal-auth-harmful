"""
Define Dataset class -- returns a dictionary with a single ECG and a single PPG
"""
from datasets.base_dual_dataset import BaseDualDataset
from datasets.dataset_metadata import DualDatasetMetadata

from torch.utils.data import Dataset
import glob
import os
import random
import numpy as np
import pickle as pkl
import mat73
import pandas as pd

from datasets.dataset_utils import sliding_window, resample, zscore, min_max_norm
from datasets.dataset_utils import filter_ecg, filter_ppg

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
        NUM_SAMPLES_TRAIN = 2178,
        NUM_SAMPLES_TEST = 594
    )

    def __init__(self, split="train", evaluate=False, fraction_time=True):
        super().__init__(split, evaluate, fraction_time)

    def get_gt_hrs(self, patient):
        base_path = datasets.get_base_path('capno')
        arr = mat73.loadmat(os.path.join(base_path, patient+"_8min.mat")) # same as signal data
        df = pd.DataFrame(arr['reference']['hr']['ecg']) # has x and y keys, for seconds and BPM estimate. x is in a weird interval (~0.6 sec)
        df.columns = ['s', 'bpm'] 
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

        FS_ORIG_ECG = datasets.get_orig_fs('capno', 'ecg')
        FS_ORIG_PPG = datasets.get_orig_fs('capno', 'ppg')

        TARGET_FS = 128 # also in metdadata
        overlap_len = 0.4 if not self.evaluate else 0.0

        for pix, p, sess, data in datasets.get_paired_data('capno', ['ecg', 'ppg'], spoof=False, split=self.split, fraction_time=(0, 0.5)):

            ecg = data['ecg']
            ppg = data['ppg']

            ecg = resample(ecg, orig_fs=FS_ORIG_ECG, target_fs=TARGET_FS)
            ecg = filter_ecg(ecg, TARGET_FS)
            ecg = zscore(ecg)
            ecg = sliding_window(ecg, fs=TARGET_FS, window_len=4, overlap_len=overlap_len)
            ecg = min_max_norm(ecg)
    
            ppg = resample(ppg, orig_fs=FS_ORIG_PPG, target_fs=TARGET_FS)
            ppg = filter_ppg(ppg, TARGET_FS)
            ppg = zscore(ppg)
            ppg = sliding_window(ppg, fs=TARGET_FS, window_len=4, overlap_len=overlap_len)
            ppg = min_max_norm(ppg)

            ECG.append(ecg)
            PPG.append(ppg)
            LABELS.append([pix for i in range(len(ecg))])
            PATIENT_NAMES.append(p)

        self.A = np.concatenate(ECG, 0)
        self.B = np.concatenate(PPG, 0)
        self.LABELS = np.concatenate(LABELS)
        self.PATIENT_NAMES = PATIENT_NAMES
        print("Loaded data of size: ", self.A.shape)




