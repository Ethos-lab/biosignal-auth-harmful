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

from scipy.signal import detrend

import neurokit2 as nk2


from datasets.dataset_utils import sliding_window, resample
from datasets.dataset_utils import filter_ecg, filter_ppg
from datasets.dataset_utils import zscore, min_max_norm

from custom_datasets import datasets   # custom package

class DualDataset(BaseDualDataset):

    METADATA = DualDatasetMetadata(
        NAME_A = 'ECG',
        NAME_B = 'RPPG',
        NUM_LEADS_A = 1,  # has 3 available
        NUM_LEADS_B = 1,
        FS_A = 128,
        FS_B = 128,
        SAMPLE_LEN = 512,
        NUM_SAMPLES_TRAIN = 357,  # 2244,  #4522 for full lenght
        NUM_SAMPLES_TEST = 84 #726
    )



    def __init__(self, split="train", evaluate=False, fraction_time=True):
        super().__init__(split, evaluate, fraction_time)


    def load_from_files(self):
        """
        From: https://physionet.org/content/bidmc/1.0.0/
        """


        ECG, RPPG = [], []
        PATIENT_LIST = []
        PATIENT_NAMES = []
        SESSIONS_A, SESSIONS_B = [], []
        self.PAT_FOR_SESS = {}

        FS_ORIG_ECG = datasets.get_orig_fs('hcitagging', 'ecg')  # 256
        FS_ORIG_RPPG = datasets.get_orig_fs('hcitagging', 'rppg')  # 60

        TARGET_FS = 128  # FS_A

        overlap_len = 0.4 if not self.evaluate else 0.0

        """
        I made this annoying for myself and want to add in multiple sesions per patient -- the current .get_paried_data 
        can't handle that because how will it know which session is which? Hmm lemme think 
        """
        fraction_time = False if not self.fraction_time else (0, 0.5)

        for pix, p, sess, data in datasets.get_paired_data('hcitagging', ['ecg', 'rppg'], spoof=False, split=self.split, fraction_time=fraction_time):

            ecg = data['ecg']
            rppg = data['rppg']

            # filter them in the same way as dalia, wesad, capno:
            ecg = nk2.ecg_clean(ecg[0], sampling_rate=FS_ORIG_ECG)
            ecg = np.expand_dims(ecg, 0)
            ecg = resample(ecg, orig_fs=FS_ORIG_ECG, target_fs=TARGET_FS)
            #ecg = filter_ecg(ecg, TARGET_FS)  # using ecg_clean to filter 
            # ADDITION: DETREND, just for this dataset because it's really bad 
            #detrend(ecg, overwrite_data=True)
            ecg = zscore(ecg)
            ecg = sliding_window(ecg, fs=TARGET_FS, window_len=4, overlap_len=overlap_len)
            ecg = min_max_norm(ecg)

            rppg = resample(rppg, orig_fs=FS_ORIG_RPPG, target_fs=TARGET_FS)  # resample each channel
            rppg = np.apply_along_axis(filter_ppg, 1, rppg)  # nb filter_ppg takes default TARGET_FS=128
            #rppg = filter_ppg(rppg, TARGET_FS)
            rppg = zscore(rppg)
            rppg = sliding_window(rppg, fs=TARGET_FS, window_len=4, overlap_len=overlap_len)
            num_rppg_sessions = rppg.shape[1]
            rppg = np.reshape(rppg, (-1, 1, 512)) # NOTE will need to change later, idk maybe
            rppg = min_max_norm(rppg)

            print("Pat/sess: ", p, sess, "ecg: ", ecg.shape[0], "rppg: ", rppg.shape[0])

            ECG.extend([ecg for _ in range(num_rppg_sessions)])
            RPPG.append(rppg) # There are 8 channels in each rppg because there are 4 sessions and 2 takes of each.
            PATIENT_LIST.append([pix for i in range(len(ecg))])
            PATIENT_NAMES.append(p)
            # Additions just for generating data 
            SESSIONS_A.extend([sess for _ in range(len(ecg))])
            SESSIONS_B.extend([sess for _ in range(len(rppg))])
            self.PAT_FOR_SESS[sess] = p



        self.A = np.concatenate(ECG)
        self.B = np.concatenate(RPPG)
        self.LABELS = np.concatenate(PATIENT_LIST)
        self.PATIENT_NAMES = PATIENT_NAMES
        self.SESSIONS_A = SESSIONS_A
        self.SESSIONS_B = SESSIONS_B
