"""
Define Dataset class -- returns a dictionary with a single ECG and a single PPG
"""

import os
import random
import wfdb
import numpy as np

from datasets.dataset_utils import sliding_window, resample
from datasets.dataset_utils import filter_ecg, filter_ppg
from datasets.dataset_utils import zscore, min_max_norm

from datasets.base_dual_dataset import BaseDualDataset
from datasets.dataset_metadata import DualDatasetMetadata

from custom_datasets import datasets

class DualDataset(BaseDualDataset):

    METADATA = DualDatasetMetadata(
      NAME_A = 'SCG',
      NAME_B = 'ECG',
      NUM_LEADS_A = 1, 
      NUM_LEADS_B = 1,  # 2 are available though
      FS_A = 128,
      FS_B = 128,
      SAMPLE_LEN = 512,
      NUM_SAMPLES_TRAIN = 47071, #10000, # 47071
      NUM_SAMPLES_TEST = 15790  #2000  # 15790
    )



    def __init__(self, split="train"):
        """
        Collected using the Biopac MP36 system. Channels 1 and 2 are leads I and II of a conventional ECG. Using just 1 though
        Channel 3 is respiratory signal obtained from a thoracic piezoresistive band
        Channel 4 is SCG using a triaxial accelerometer
        Each channel sampled at 5000 Hz
        Files that start with b (b001 to b020) are resting (basal) state
        Files that start with m (m001 to m020) are when they're listening to classical music.
        For now use the basal rate ones
        """
        super().__init__(split)


    def load_from_files(self):

        SCG, ECG = [], []
        LABELS = []
        FS_ECG = datasets.get_orig_fs('cebsdb', 'ecg')  # 7000
        FS_SCG = datasets.get_orig_fs('cebsdb', 'scg')  # 7000
        TARGET_FS = 128 # let's just make everything 128 hz, why not


        for pix, p, data in datasets.get_paired_data('cebsdb', ['ecg', 'scg'], spoof=False, split=self.split, fraction_time=(0, 0.5)):

            scg = data['ecg']
            ecg = data['ecg']            

            # same steps as ecg/ppg in cardiogan:
            scg = resample(scg, orig_fs=FS, target_fs=TARGET_FS)
            scg = filter_ecg(scg, TARGET_FS)  # TODO filering same as ecg, maybe should do ppg
            scg = zscore(scg)
            scg = sliding_window(scg, fs=TARGET_FS, window_len=4, overlap_len=0.1)
            scg = min_max_norm(scg)


            ecg = np.expand_dims(ecg, 0)
            ecg = resample(ecg, orig_fs=FS, target_fs=TARGET_FS)
            ecg = zscore(ecg)
            ecg = sliding_window(ecg, fs=TARGET_FS, window_len=4, overlap_len=0.1)
            ecg = min_max_norm(ecg)

            SCG.append(scg)
            ECG.append(ecg)
            LABELS.append([pix for i in range(len(scg))])

        self.A = np.concatenate(SCG)
        self.B = np.concatenate(ECG)
        self.LABELS = np.concatenate(LABELS)
        print("Loaded data of size: ", self.A.shape)


