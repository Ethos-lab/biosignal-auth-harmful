"""
Define Dataset class -- returns a dictionary with a pair of data
"""

from torch.utils.data import Dataset
from dataclasses import dataclass
import random
import numpy as np
from datasets.dataset_metadata import DualDatasetMetadata


class BaseDualDataset(Dataset):

    # Metadata dict so we have an easy way of accessing the data types, fs's etc
    METADATA = DualDatasetMetadata(
        NAME_A = 'ECG',
        NAME_B = 'ECG',
        NUM_LEADS_A = 1,
        NUM_LEADS_B = 1,
        FS_A = 128,
        FS_B = 128,
        SAMPLE_LEN = 512,
        NUM_SAMPLES_TRAIN = 0,
        NUM_SAMPLES_TEST = 0
    )

    def __init__(self, split="train", evaluate=False, fraction_time=True):
        super().__init__()
        self.split = split
        self.num_samples = self.METADATA.NUM_SAMPLES_TRAIN if split == "train" else self.METADATA.NUM_SAMPLES_TEST
        self.evaluate = evaluate
        self.fraction_time = fraction_time
        self.load_from_files()
        assert hasattr(self, 'A') and hasattr(self, 'B') and hasattr(self, 'LABELS')  # these are the things that need to be set in load_from_files
        print(f'Loaded {self.split} dataset with {self.A.shape} samples')

    def get_gt_hrs(self, patient):
        """
        Helper for manually evaluating performance (ecg/ppg mainly). Not needed for training/testing.
        """
        pass

    def load_from_files(self):
        """
        Needs to load save self.A and self.B as numpy arrays
        """
        raise NotImplementedError("Base class doesnt have this implemented")

    def get_pair(self, index):
        ''' Return a paired set of data {"A": data, "B": data} or an empty dictionary if the dataset isn't paired
            This is used for debugging and plotting
        '''
        return {'A': self.A[index].astype(np.float32),
                'B': self.B[index].astype(np.float32)
                }

    def __getitem__(self, index):
        " Gets the [index]'th item of EGM, but gets a random item from ECG. We dont want the network to memorize the same mappings "
        index_random = random.randint(0, len(self.A)-1)
        ix_A = self.LABELS[index_random]  # subject number for A
        ix_B = self.LABELS[index] # subject number for B
        return {"A": self.A[index_random].astype(np.float32),
                "B": self.B[index].astype(np.float32),
                "LABEL_A": ix_A,
                "LABEL_B": ix_B
                }

    def __len__(self):
        return min(self.A.shape[0], self.B.shape[0]) 
