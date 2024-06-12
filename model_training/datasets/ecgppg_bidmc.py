"""
Define Dataset class -- returns a dictionary with a single ECG and a single PPG
"""
from datasets.dataset_metadata import DualDatasetMetadata
from datasets.base_dual_dataset import BaseDualDataset
import glob
import os
import random
import numpy as np
import pickle as pkl

import pandas as pd


from datasets.dataset_utils import sliding_window, resample
from datasets.dataset_utils import filter_ecg, filter_ppg
from datasets.dataset_utils import zscore, min_max_norm

from custom_datasets import datasets   # custom package

class DualDataset(BaseDualDataset):

    METADATA = DualDatasetMetadata(
        NAME_A = 'ECG',
        NAME_B = 'PPG',
        NUM_LEADS_A = 1,  # has 3 available
        NUM_LEADS_B = 1,
        FS_A = 128,
        FS_B = 128,
        SAMPLE_LEN = 512,
        NUM_SAMPLES_TRAIN = 2244,  #4522 for full lenght
        NUM_SAMPLES_TEST = 726
    )



    def __init__(self, split="train", evaluate=False, fraction_time=True):
        super().__init__(split, evaluate, fraction_time)

    def get_gt_hrs(self, patient):
        base_path = datasets.get_base_path('bidmc') # should set on class instead of calling each time but whatever
        gt_fn = os.path.join(base_path, "bidmc_"+patient+"_Numerics.csv")
        numerics = pd.read_csv(gt_fn)
        numerics.columns = [c.strip() for c in numerics.columns]
        hrs = numerics[['Time [s]', 'HR']]
        hrs.columns = ['s', 'bpm']
        return hrs  # this is a df with a HR (in bpm) calculated each second

    def load_from_files(self):
        """
        From: https://physionet.org/content/bidmc/1.0.0/
        """

        num_samples = self.METADATA.NUM_SAMPLES_TRAIN if self.split == "train" else self.METADATA.NUM_SAMPLES_TEST

        if self.split == 'all':  
            raise NotImplementedError # why are we doign this again? 


        ECG, PPG = [], []
        PATIENT_LIST = []
        PATIENT_NAMES = []

        FS_ORIG_ECG = datasets.get_orig_fs('bidmc', 'ecg') # 125
        FS_ORIG_PPG = datasets.get_orig_fs('bidmc', 'ppg')

        TARGET_FS = 128  # FS_A

        overlap_len = 0.4 if not self.evaluate else 0.0

        for pix, p, sess, data in datasets.get_paired_data('bidmc', ['ecg', 'ppg'], spoof=False, split=self.split, fraction_time=(0, 0.5)):

            ecg = data['ecg']
            ppg = data['ppg']

            # filter them in the same way as dalia, wesad, capno:
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
            PATIENT_LIST.append([pix for i in range(len(ecg))])
            PATIENT_NAMES.append(p)



        self.A = np.concatenate(ECG)
        self.B = np.concatenate(PPG)
        self.LABELS = np.concatenate(PATIENT_LIST)
        self.PATIENT_NAMES = PATIENT_NAMES
