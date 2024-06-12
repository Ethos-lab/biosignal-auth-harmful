from torch.utils.data import Dataset
import numpy as np

from custom_datasets import datasets, processing

class TrainDataset(Dataset):
    def __init__(self, split, num_heartbeats_per_user=None, both_bcg_ecg=False, T=15, spoof_name='original'):
        assert not num_heartbeats_per_user

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

        spoof = spoof_name != 'original'
        fs_ecg = datasets.get_orig_fs('bedbased', datatype='ecg', spoof=False)
        fs_bcg = datasets.get_orig_fs('bedbased', datatype='bcg', spoof=spoof, spoof_name=spoof_name)  # may be true or not
        
        for pix, patient, data in datasets.get_orig_spoof_paired_data('bedbased', 'ecg', 'bcg', spoof_name=spoof_name, split='all', fraction_time=fraction_time):
            ecg, bcg = data['ecg'], data['bcg']

            bcg = processing.filter_signal(bcg, freqs=[1.0, 30.0], fs=fs_bcg)
            bcg = processing.zscore(bcg)

            bcg = processing.resample_signal(bcg, fs_bcg, 500)
            ecg = processing.resample_signal(ecg, fs_ecg, 500)
            rlocs = processing.get_ecg_rpeaks_hamilton(ecg, 500)

            segmented = self.segment_bcg(bcg, rlocs)
            if both_bcg_ecg:
                segmented_ecg = self.segment_ecg(ecg, rlocs)

            if not both_bcg_ecg:
                group_segments = self.group_segments(segmented)  # from indiv hbs to the T-length hbs we need
            else:   
                group_segments = self.group_segments_both(segmented, segmented_ecg)

            
            self.DATA.extend(group_segments)
            self.PATIENT_IX.extend([pix for i in range(len(group_segments))])

            self.SAMPLE_LEN = group_segments[0].shape[1]

            print(f"Processed data for patient: {patient}, num heartbeats: {len(segmented)}")

    def group_segments(self, segments):
        # extend 15 more
        group_segs = []
        for si in range(len(segments)):
            if si + self.T >= len(segments):  break
            group_segs.append(np.concatenate(segments[si:si+self.T], 0))
        return group_segs

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
