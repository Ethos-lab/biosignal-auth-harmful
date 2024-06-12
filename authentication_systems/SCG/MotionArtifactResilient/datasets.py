import numpy as np

from custom_datasets import datasets, processing

import signal_processing

class TrainDataset:
    def __init__(self, split, spoof_name='original'):

        if split == 'train':
            fraction_time = (0.5, 0.625)
        elif split == 'test':
            fraction_time =  (0.625, 0.75)
        elif split == 'eval':
            fraction_time = (0.75, 1.0)
        else:
            raise ValueError

        self.data, self.users = [], []

        spoof = spoof_name != 'original'
        RESAMPLE_FS=512

        fs_ecg = datasets.get_orig_fs('cebsdb', datatype='ecg', spoof=False)
        fs_scg = datasets.get_orig_fs('cebsdb', datatype='scg', spoof=spoof, spoof_name=spoof_name)
        
        for pix, patient, data in datasets.get_orig_spoof_paired_data('cebsdb', 'ecg', 'scg', spoof_name=spoof_name, split='all', fraction_time=fraction_time):
            ecg, scg = data['ecg'], data['scg']

            # Going to resample to 128 to make it easy for us, since they dont care
            scg = processing.resample_signal(scg, fs_scg, RESAMPLE_FS)[0]
            scg = signal_processing.filter(scg, RESAMPLE_FS)

            _, rpeaks = processing.resample_get_rpeaks(ecg, orig_fs=fs_ecg, resample_to=RESAMPLE_FS) 

            ao_peaks = signal_processing.get_AO_peaks(scg, rpeaks, RESAMPLE_FS)  # time-shifted from rpeaks

            sigs = signal_processing.segment(scg, ao_peaks, fs=RESAMPLE_FS)  # into individual cardiac cycles

            # Let's see how this helps, because spoofed is normalized (-1 1)
            sigs = signal_processing.min_max_norm(sigs)
            # sigs = signal_processing.zscore(sigs)

            sigs = signal_processing.average_sigs(sigs)

            # Note: for dep elearning, "we compress the wavelet transformed SCG image to a size of 80x80"

            self.data.extend(sigs)

            self.users.extend([pix for x in range(len(sigs))])

            #print(f"Processed data for user: {user}, shape: {len(sigs)}")
            print(".", end='', flush=True)

        self.data = np.stack(self.data, 0)  # now 2D
        print('', flush=True)

    def __len__(self):
        return len(self.users)
