# Implementing Key To Your Heart approach: https://arxiv.org/pdf/1906.09181.pdf
# Not the most complicated approach, but it's decently recent, and they evaluate KardiaMobile traces. 
# SVM-based, after standard signal processing
import argparse
import os
from signal_processing import SignalProcessor
from sklearn.decomposition import PCA
import mat73
import numpy as np

from custom_datasets import datasets
from collections import defaultdict

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Key To Your Heart: Loads, prepares, splits data, writes to big_hdd')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--spoof_name', type=str, default='original')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help="Note we fixed this - it does on time, not on split split")
    parser.add_argument('--dryrun', action='store_true')    
    args = parser.parse_args()

    spoof = args.spoof_name != 'original'
    split = args.split

    if split == 'train':
        fraction_time = (0.5, 0.75)
    elif split == 'test':
        fraction_time = (0.75, 1.0)
    else:
        raise ValueError

    # note these can be obtained from my module
    fs = datasets.get_orig_fs(args.dataset, spoof=spoof, spoof_name=args.spoof_name)

    # Need to separate the data we used to train the cyclegan from test.... Dont want to test this using spoofed data that was generated
    # for a subject within the cyclegan training split.. 
    print('Split: ', split)
    print("Fraction time: ", fraction_time)

    write_path = 'processed/data/write/path'
    if not args.dryrun:  os.makedirs(write_path, exist_ok=True)

    data_generator = datasets.get_data(args.dataset, 'ecg', spoof=spoof, spoof_name=args.spoof_name, split='all', fraction_time=fraction_time)

    num_samps_total = 0
    num_pats_total = 0

    # Addition for handling multisession data 
    patient_data = defaultdict(list)
    for pix, patient, sess, ecg in data_generator:
        processor = SignalProcessor(fs)
        signals = processor.process(ecg)
        patient_data[patient].extend(signals)

    for patient, signals in patient_data.items():

        # Statistics just to print out and save
        num_samps_for_pat = len(signals)
        num_samps_total += num_samps_for_pat

        # 1. Dimensionality reduction and feature extraction
        pca = PCA(n_components=25)
        features = pca.fit_transform(signals)

        fn = os.path.join(write_path, patient + '.npy')
        if not args.dryrun:
            np.save(fn, features)
            print(f"Wrote {num_samps_for_pat} for patient {patient} to: ", fn)
        else:
            print(f"Would write {num_samps_for_pat} for patient {patient} to: ", fn)

            
    print()
    print("Total num patients: ", len(patient_data))
    print("Total num samps: ", num_samps_total)
        
