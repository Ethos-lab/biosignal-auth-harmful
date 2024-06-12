import numpy as np
import random

from custom_datasets import datasets, processing
from torch.utils.data import Dataset
import signal_processing as sp

class MulticlassDataset(Dataset):
    def __init__(self, dataset, split,  fs=64, spoof_name='original', channels=['ZP','ZT'], augmentation=True):
        if split == 'train':
            fraction_time = (0.5, 0.75)
        elif split == 'test':
            raise NotImplementedError # Why are we here? 
        elif split == 'eval':
            fraction_time = (0.75, 1.0)
        else:
            raise ValueError

        self.channels = channels # just for saving later
        self.num_channels = len(channels)

        self.data, self.users = [], []

        RESAMPLE_FS = 128  # to be consistent 
        fs = datasets.get_orig_fs(dataset, 'ppg', spoof=spoof_name!='original', spoof_name=spoof_name)  # orig fs
        for uix, user, ppg in datasets.get_data(dataset, 'ppg', split='all', spoof=spoof_name != 'original', spoof_name=spoof_name, fraction_time=fraction_time):

            if fs != RESAMPLE_FS:  ppg = processing.resample_signal(ppg, fs, RESAMPLE_FS)

            # Paper uses 100Hz or 300Hz, so our 128Hz isn't bad
            ppg = sp.min_max_norm(ppg)
            ppg = sp.filter(ppg, RESAMPLE_FS)
            segs = sp.segmentation(ppg, RESAMPLE_FS)
            
            if augmentation:
                segs = sp.data_aug(segs, noise_type='combination')  # first add noise

            # Hardcode: doing DTW+ZT 
            transformed_segs = []
            for ch in channels:
                transformed = sp.stretching(segs, RESAMPLE_FS, ch)
                transformed_segs.append(transformed)
            stacked = np.stack(transformed_segs, 1)

            self.data.extend(stacked)
            self.users.extend([uix for x in range(stacked.shape[0])])

            #print(f"Processed data for user: {user} for uix: {uix}. Shape: {stacked.shape}")

        # Data shape now:
        #<indiv-samples>x2x<length-per-sample>
        self.data = np.array(self.data)
        self.num_users = len(set(self.users))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        return self.data[ix], self.users[ix]

    def get_user_ixs(self, user):
        # Returns subset ixs for user
        subset = [i for i, u in enumerate(self.users) if u == user]
        return subset

    def get_other_user_ixs(self, user):
        # Returns subset ixs for user
        subset = [i for i, u in enumerate(self.users) if u != user]
        return subset

class BinaryDataset(Dataset):
    def __init__(self, fulldataset, user, neg_multiplier=2.0, baseline=False, eval=False):  # just trying it
        # Wrapper around MulticlassDataset where total len is 2*num_for_this_user. So prob we get me is 1/neg_multipler
        self.dset = fulldataset
        self.user = user
        self.neg_multiplier = neg_multiplier  # rand prob at 1/neg_multiplier
        if baseline:
            num_for_user = len(self.dset.get_user_ixs(user))
            self.user_ixs = self.dset.get_other_user_ixs(user)
            random.shuffle(self.user_ixs)
            self.user_ixs = self.user_ixs[:num_for_user]  # so it's the same number available 
        else:
            self.user_ixs = self.dset.get_user_ixs(user)
        self.someone_else_ixs = list(set(list(range(len(self.dset)))).difference(self.user_ixs))  # the remaining
        random.shuffle(self.someone_else_ixs)  # so it's not all the same person

        self.baseline = baseline
        self.eval = eval

    def __len__(self):
        if self.neg_multiplier < 0:  # just give all the data
            return len(self.dset)
        else:
            return int(self.neg_multiplier*len(self.user_ixs))


    def __getitem__(self, ix):
        if self.eval:
            return self.dset[ix][0], True  # with neg_multiplier=1.0, this willa ctually be the case
        else:
            prob_me = random.random() < (1/self.neg_multiplier)
            if prob_me:
                ix = ix % len(self.user_ixs)
                my_ix = self.user_ixs[ix]
                return self.dset[my_ix][0], True  
            else:
                someone_else = random.choice(self.someone_else_ixs)
                return self.dset[someone_else][0], False
