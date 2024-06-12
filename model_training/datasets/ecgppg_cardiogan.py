"""
Define Dataset class -- returns a dictionary with a single ECG and a single PPG
"""
from datasets.base_dual_dataset import BaseDualDataset
from datasets.dataset_metadata import DualDatasetMetadata
import importlib
import numpy as np
from multiprocessing import Pool

class DualDataset(BaseDualDataset):
    """
    Loads DALIA, BIDMC, CAPNO, WESAD (Cardiogan dataset combines all 4)
    Resamples each (both ECG and PPG) to 128 Hz
    Then band-pass FIR filter with a pass-band of 3Hz and a stop-band of 45Hz (ECG) and 1Hz/8Hz (PPG)
    Then Z-scoring
    Then normalized into 4-second windows (512 samples each because 128Hz x 4 seconds) with 10% overlap "to avoid missing any peaks"
    Then min-max normalization
    """

    METADATA = DualDatasetMetadata(
        NAME_A = "ECG",
        NAME_B = "PPG",
        NUM_LEADS_A = 1,
        NUM_LEADS_B = 1,
        FS_A = 128,
        FS_B = 128,
        SAMPLE_LEN = 512,
        NUM_SAMPLES_TRAIN = 28548,
        NUM_SAMPLES_TEST = 7276
    )

    def __init__(self, split="train", evaluate=False, fraction_time=True):
        super().__init__(split, evaluate, fraction_time)

    def parallel_load_dataset(self, name, pix_offset):
        dmodule = importlib.import_module('datasets.ecgppg_'+name.lower())
        print("Loading: ", name, self.split, self.evaluate)
        dset = dmodule.DualDataset(self.split, self.evaluate)
        self.A.append(dset.A)
        self.B.append(dset.B)
        fixed_labels = [l+pix_offset for l in dset.LABELS]
        self.LABELS.extend(fixed_labels)
        label_offset = max(self.LABELS)+1
        return label_offset


    def load_from_files(self):

        num_samples = self.METADATA.NUM_SAMPLES_TRAIN if self.split == "train" else self.METADATA.NUM_SAMPLES_TEST
        datasets = ['dalia', 'capno', 'bidmc', 'wesad']


        self.A = []
        self.B = []
        self.LABELS = []

        #with Pool(4) as p:
        #    p.map(self.parallel_load_dataset, datasets)

        label_offset = 0 # person identity index for contrastive; dont want all to start at 0
        for d in datasets:
            label_offset = self.parallel_load_dataset(d, label_offset)

        self.A = np.concatenate(self.A)
        self.B = np.concatenate(self.B)
        self.LABELS = np.array(self.LABELS)

        print("Loaded all cardiogan4 data: ", self.A.shape)
