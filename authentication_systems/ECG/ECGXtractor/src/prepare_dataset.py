"""
 In the version of ECGXtractor I cloned, when you wanted to prepare a new datasets, you'd have to create a new directory within src
and copy over build_segments.py (which loads raw data, filters, segments, etc,writes out to a given location) and create_datasets.py
(which harcodes which segments are in training/testing/validation folds within json files specified in configs). 

I did that manually for a while but was not scalable. 

Purpose of this is a extendable that combines build_segments and create_datasets for any dataset
"""

import argparse 
from custom_datasets import datasets, processing

import numpy as np
import tqdm
import pandas as pd
import neurokit2 as nk2
import os
import glob
import json
import pickle as pkl 
import glob
import scipy.signal as scipy_signal
import operator
import random
import time
from math import floor
from collections import defaultdict


def resample(signal, orig_fs = 700, target_fs=128):
    ''' Reamples using scipy.signal.resample, which uses fourier interpolation.
        ECGXtractor needs 500Hz
        Make sure input signal is shape num_leads x num_sample '''
    num_samp_orig = signal.shape[1]
    num_samp_target = round( (target_fs * num_samp_orig) / orig_fs )
    resized = scipy_signal.resample(signal, num_samp_target, axis=1)
    return resized


def get_peaks(base_path, pats):
    dict_peaks = {}
    for p in pats:
        dict_peaks[p] = {}
        path = os.path.join(base_path, p)
        sessions = [name for name in os.listdir(path) if not os.path.isfile(os.path.join(path, name))]
        for s in sessions:
            path_session = os.path.join(base_path, p, s)
            peaks = [os.path.join(s, name) for name in os.listdir(path_session) if os.path.isfile(os.path.join(path_session, name))]
            #peaks = [s + '/' + name for name in os.listdir(path_session) if
            #         os.path.isfile(os.path.join(path_session, name))]
            dict_peaks[p][s] = peaks
    return dict_peaks

def build_advanced_segment(sn, best_n):
    avg_segment = np.mean(sn, axis=0)

    distance = [np.linalg.norm(s - avg_segment) for s in sn] 
    idx_distance = list(enumerate(distance))
    top_n = [t[0] for t in sorted(idx_distance, key=operator.itemgetter(1))[:best_n]]

    segments_selected = [s for i, s in enumerate(sn) if i in top_n]
    result = np.mean(segments_selected, axis=0)

    return result

def keep_peaks_sessions(dict_peaks, sessions, verification):
    # session is a string and can be: 'single', 'multi'
    # verification is a boolean
    kept_peaks = {}
    for k, v in dict_peaks.items():
        kept_peaks[k] = {}
        if sessions == 'single':
            single = v[list(v.keys())[0 if len(v)==1 else 1]]
            single = [s for s in single if not (s.split('/')[1][:8] == 'template' or s.split('/')[1][:2] == 'hb')]
            kept_peaks[k]['single'] = single
        elif sessions == 'multi':
            #assert (len(v) > 1)  # Well sometimes we only have one
            template = v[list(v.keys())[0]]
            template = [t for t in template if t.split('/')[1][:8] == 'template']
            kept_peaks[k]['template'] = template
            sample = v[list(v.keys())[1]]
            sample = [s for s in sample if s.split('/')[1][:8] == 'template']
            kept_peaks[k]['sample'] = sample
        elif sessions == 'multi-custom':
            # Treat multiple sessions as just more data, since hcitaggng has lots of sessions but none are that long
            kept_peaks[k]['single'] = []
            for sess, files in v.items():
                kept_peaks[k]['single'].extend(files)
    return kept_peaks


def get_data_from_users_for_verification(kept_peaks, user_list):
    data = []
    if 'single' in list(kept_peaks[list(kept_peaks.keys())[0]].keys()):
        for u in user_list:
            data = data + [{'ecg': vv, 'usercode': u} for vv in kept_peaks[u]['single']]
    elif 'template' in list(kept_peaks[list(kept_peaks.keys())[0]].keys()):
        for u in user_list:
            data = data + [{'ecg': kept_peaks[u]['template'][0], 'usercode': u, 'function': 'template'}]
            data = data + [{'ecg': kept_peaks[u]['sample'][0], 'usercode': u, 'function': 'sample'}]
    return data

def create_ds(kept_peaks, split_pct, mode):
    # split_pct is a dict with keys 'train', 'val', 'test' and the sum of values is 1.
    # mode is a string and can be: 'verification', 'identification'
    if mode == 'verification':
        users = list(kept_peaks.keys())
        random.shuffle(users)
        assert(split_pct['train'] + split_pct['val'] + split_pct['test'] == 1)
        train_users = users[:int(split_pct['train']*len(users))]
        val_users = users[int(split_pct['train']*len(users)):int((split_pct['train']+split_pct['val'])*len(users))]
        test_users = users[int((split_pct['train']+split_pct['val'])*len(users)):]
        training_ds = get_data_from_users_for_verification(kept_peaks, train_users)
        validation_ds = get_data_from_users_for_verification(kept_peaks, val_users)
        testing_ds = get_data_from_users_for_verification(kept_peaks, test_users)
    else:
        raise ValueError('Got rid of verification')
    return training_ds, validation_ds, testing_ds

def write_json(ds, fn):
    with open(fn, 'w') as fid:
        for tr in ds:
            tr['ecg'] = tr['usercode'] + '/' + tr['ecg']
            json.dump(tr, fid)
            fid.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Runs both build_segments and create_datasets for given dataset. When training, loads the train patients and splits into train/val. When eval, loads the test patients and puts all in val. Writes to name/type/split directory, so the train split should have both train/eval but the test split only has eval')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--eval', action='store_true', help='For spoof datasets, splits and assigns val to everything. Also (0.75, 1.0). Otherwise (0.5, 0.75)')
    parser.add_argument('--spoof_name', type=str, default='original')
    parser.add_argument('--dryrun', action='store_true')
    args = parser.parse_args()


    t0 = time.time()

    # Hardcoded params
    consecutive_heartbeats = 10  # Using the 'template' approach with 10 consecutive_heartbeats
    segment_length = 400
    segment_back = 160  # these are really strange numbers but okay
    max_windows_per_patient = 1000  # wont have this problem tho
    best_number = 5
    word = 'ss'  # summary segment, for naming write out files


    print("BUILD_SEGMENTS step:  Loading, segmenting, writing out new files")

    split = 'all'
    split_name = 'eval' if args.eval else 'train'
    fraction_time = (0.75, 1.0) if args.eval else (0.5, 0.75)

    yes_spoof = args.spoof_name != 'original'
    fs = datasets.get_orig_fs(args.dataset, 'ecg', spoof=yes_spoof, spoof_name=args.spoof_name)
    patients = datasets.get_list_of_patients(args.dataset, spoof=yes_spoof, spoof_name=args.spoof_name, split=split)

    # write_path = ....ecgxtractor... bidmc/original/train or bidmc/original/test, indicating the
    # patient split used. and then for a spoof one for exmaple, bidmc/my_cardiogan/test only 
    write_path = os.path.join('/media', 'big_hdd', 'kveena', 'ecgxtractor', 'datasets', args.dataset, args.spoof_name, split_name)

    NUM_SEGS_PER_PATIENT = defaultdict(int)  # to just print out at the end for our fyi
    SEGS_PER_PATIENT = defaultdict(list)

    if os.path.exists(write_path) and not args.dryrun:
        rewrite = input(f"Folder {write_path} already exists. Continuing will overwrite contents. y/n?")
        if rewrite.lower() in ["y", "yes"]:
            import shutil
            shutil.rmtree(write_path)
        else:
            print("Quitting")
            exit(0)

    fs_orig = datasets.get_orig_fs(args.dataset, 'ecg', yes_spoof, args.spoof_name)

    PAT_SESSION_IXS = defaultdict(lambda: -1) # increment when we see a new session for patient
    for pix, patient, sess, ecg in datasets.get_data(args.dataset, 'ecg', spoof=yes_spoof, spoof_name=args.spoof_name, fraction_time=fraction_time):
        if not args.dryrun:
            if not os.path.exists(os.path.join(write_path, patient)):  os.makedirs(os.path.join(write_path, patient))

        PAT_SESSION_IXS[patient] += 1


        if args.dataset == 'hcitagging':
            # Highpass filter and peak detection 
            cleaned = nk2.ecg_clean(ecg[0], sampling_rate=fs_orig)
            _, rpeaks = nk2.ecg_peaks(cleaned, sampling_rate=fs_orig, correct_artifacts=True)  # default 'neurokit'  method instead of hamilton 
            rpeaks = rpeaks['ECG_R_Peaks']
            cleaned = np.expand_dims(cleaned, 0)

            # Then normalization
            ecg = processing.zscore(cleaned)
            ecg = processing.min_max_norm(ecg)

            # Resample the signal (and rpeaks)
            ecg = processing.resample_signal(ecg, fs_orig, 500)
            rpeaks = processing.resample_rpeak_arr(rpeaks, fs_orig, 500) 

            ecg = ecg.T # num_samples x 1
            print("Done with ", patient, sess)
        else:
            ecg = processing.zscore(ecg) 
            ecg = processing.min_max_norm(ecg)  # To put everythign on the same scale and that expected by autoencoder

            # Reample to 500Hz, and scale the peaks as well
            ecg, rpeaks = processing.resample_get_rpeaks(ecg, fs_orig, resample_to=500)
            ecg = ecg.T  # now num_samples x 1

        # Segment into windows of 400 samples, centered around each peak 
        # Each segment should have ~400 samples; np.diff(rpeaks) should be around 400 I think? 
        segments = [ecg[r - segment_back: r + segment_length - segment_back, :] for r in rpeaks[1:-1]]

        segments = [s for s in segments if s.shape == (400, 1)]  # unclear why it wouldn't be but okay
    
        cons_segments = [segments[i:i + consecutive_heartbeats] for i in range(0, len(segments) - consecutive_heartbeats + 1, consecutive_heartbeats)]


        if len(cons_segments) > max_windows_per_patient:
            cons_segments = cons_segments[:max_windows_per_patient]


        # Build the templates and write out
        NUM_SEGS_PER_PATIENT[patient] += len(cons_segments)
        session_ix = PAT_SESSION_IXS[patient]
        SEGS_PER_PATIENT[patient].append(cons_segments) # Fix for multisession ('session_ix' is 'sample')

    # Then iterate again and write 
    for patient, session_cons_segment in SEGS_PER_PATIENT.items():
        for sample, cons_segment in enumerate(session_cons_segment):
            sample = str(sample)
            if not args.dryrun:
                if not os.path.exists(os.path.join(write_path, patient, sample)): os.makedirs(os.path.join(write_path, patient, sample))
            for i, sn in enumerate(cons_segment):
                result = build_advanced_segment(sn, best_number)
                df = pd.DataFrame(result)
                csvfn = os.path.join(write_path, patient, sample, word + "_" + "{:03n}".format(i) + ".csv")
                if not args.dryrun:
                    print('Writing: ', csvfn)
                    df.to_csv(csvfn, index=False, header=False)
                else:
                    print('Would write to: ', csvfn)




    print('CREATE_DATASETS step: assigning dataset folds and writing to csv files', flush=True)
    json_write_path = os.path.join('datasets', 'jsons', args.dataset, args.spoof_name, split_name)
    if not args.dryrun:
        # Create a new folder within datasets
        os.makedirs(json_write_path, exist_ok=True)

    # Only bothering to support verification here, with a split of 70% train, 30% val, 0 test
    if args.eval:
        print('Eval mode: saving everything as val')
        split_pct = {'train': 0., 'val': 1.0, 'test': 0}
    else:
        split_pct = {'train': 0.7, 'val': 0.3, 'test': 0} 

    peaks = get_peaks(write_path, patients)

    # Okay what if I just dont do this
    session_type = 'single' if not args.dataset == 'hcitagging' else 'multi-custom'
    kept_peaks = keep_peaks_sessions(peaks, session_type, True)  # ? session, verification


    train_ds, val_ds, _ = create_ds(kept_peaks, split_pct, 'verification')

    for ds, dsplit in zip([train_ds, val_ds], ['train', 'val']):
        fn = os.path.join(json_write_path, f'{dsplit}.json')
        if not args.dryrun:
            print("Writing json: ", fn)
            write_json(ds, fn)
        else:
            print('Would write json to: ', fn)



    print(f"COMPLETE, took: {(time.time()-t0)/60:.2f}m")
    print("Num segs created per patient: ")
    for k, v in NUM_SEGS_PER_PATIENT.items():
        print(f"Pat {k}: {v}")



