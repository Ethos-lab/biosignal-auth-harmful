"""
Helpers for loading, getting info, on raw, original datasets used throughout the entire project. 
Functions for getting read path, fs; generators for getting patient_name, sig_sample  (num_leads x num_samples). Main usage:

generator = get_data(dataset_name, data_type, spoof=False, spoof_name=None, split='train', fraction_time=(0, 0.5), session='1')
subject_ix, subject_name, session_name, data = next(generator)
    

"""
import os
import glob
import pandas as pd
import numpy as np
from scipy.io import loadmat  # for mat versions under 73
import mat73  # for mat versions >= 73
import pickle as pkl
import json
from pyedflib import highlevel
import wfdb
import neurokit2 as nk2
from itertools import chain

from custom_datasets.properties import get_props
from custom_datasets import processing


# Some datasets (i.e. the video ones) have multiples sessions of collection. For those that don't assign '0'
SESSION_DEFAULT = '0'

def get_rpeaks_from_ecg(ecg, fs): 
    cleaned = nk2.ecg_clean(ecg, sampling_rate=fs, method='hamilton2002')
    _, res = nk2.ecg_peaks(cleaned, sampling_rate=fs, method='hamilton2002')
    peaks = res['ECG_R_Peaks']
    return peaks
    
def get_read_path(dataset, spoof=False, spoof_name=None):
    props = get_props(dataset, spoof, spoof_name)
    return props.READ_PATH, props.OPTIONAL_ALT_READ_PATH

def get_orig_fs(dataset, datatype='ecg', spoof=False, spoof_name=None):
    ''' The orig ecg samplign frequency of the data read in. 
        Example  125 (orig) -> 128 (cyclegan) -> 500 (ecgxtractor)
    '''
    props = get_props(dataset, spoof, spoof_name)
    return props.FS[datatype]

def get_list_of_patients(dataset, spoof=False, spoof_name=None, split='all'):
    if dataset == 'cardiogan4':  # ugh why did i make my life so difficult
        dnames = ['bidmc', 'capno', 'dalia', 'wesad']
        pats = [get_list_of_patients(dn, spoof, spoof_name, split) for dn in dnames]
        all_pats = []
        for dn, dpats in zip(dnames, pats):
            renamed_dpats = [dn+'_'+p for p in dpats]
            all_pats.extend(renamed_dpats)
        return all_pats

    props = get_props(dataset, spoof, spoof_name)
    if split == 'train':
        patients = props.TRAIN_SET
    elif split == 'test':
        patients = props.TEST_SET
    elif split == 'all':
        patients = props.TRAIN_SET + props.TEST_SET
    else:
        raise NotImplementedError("Unknown type of split: ", split)
    return patients

def get_list_of_sessions(dataset, subject):

    base_path, optional_alt_base_path = get_read_path(dataset, False, None)
    if dataset == 'hcitagging':
        with open(os.path.join(base_path, 'subject_session_mapping.pkl'), 'rb') as fp:
            session_mapping = pkl.load(fp)
        return session_mapping[subject]
    else:
        raise NotImplementedError 


def get_paired_data(dataset, data_types, spoof=False, spoof_name=None, split='all', persons=None, fraction_time=(0, 0.5), sessions=None):
    '''
    Instead of returning a tuple (index-num, patient-name, ndarray of (num_leads, num_samples), returns tuple with dicts for each data type. Specifically:  
        (index_number, patient_name, {'data-type1': (num_leads, num_samples), 'data-type2': (num_leads, num_samples), )

    if persons is provided, ignores the split

    IMPORTANT! Introducing 'fraction_time' which is a tuple designating what time fraction of each trace to get. 

    '''
    assert isinstance(data_types, list)
    assert persons is None or isinstance(persons, list)

    # Annoyingly cant think of a neater way to do this other than to break the existing generator and just create a new one

    if not persons:  persons = get_list_of_patients(dataset, spoof, spoof_name, split)
   
    """ 
    for pix, person in enumerate(persons):

        expand_gen = [get_data(dataset, dt, spoof, spoof_name, split, [person], fraction_time) for dt in data_types]
        data_dict = {dt: next(eg)[-1] for (dt, eg) in zip(data_types, expand_gen)}
        yield (pix, person, data_dict)
    """
    gen_list = [get_data(dataset, dt, spoof, spoof_name, split, persons, fraction_time, sessions) for dt in data_types]
    counter = -1
    done = False
    while not done:
        try:
            iteration = [next(gen) for gen in gen_list] 
            pat_name = iteration[0][1] # should be the same for all, but whatever 
            sess = iteration[0][2]
            if iteration[0][2] != iteration[1][2]: # sessions not same, test
                print("Warning: data not paired exactly (see datasets.py)")
            data_dict = {dt: it[-1] for (dt, it) in zip(data_types, iteration)}
            counter += 1
            yield counter, pat_name, sess, data_dict  # hope that works
        except StopIteration:
            done = True
   
def get_orig_spoof_paired_data(dataset, orig_datatype, spoof_datatype, spoof_name=None, split='all', persons=None, fraction_time=(0, 0.5)):
    '''
    Special case for SCG/HswWaveletTransform honestly... it needs the paired ecg to estimate peaks 
    ''' 
    if not persons:  persons = get_list_of_patients(dataset, spoof=True, spoof_name=spoof_name, split=split)

    orig_gen = get_data(dataset, spoof_datatype, spoof_name !='original', spoof_name, split, persons, fraction_time)
    spoof_gen = get_data(dataset, spoof_datatype, spoof_name!='original', spoof_name, split, persons, fraction_time)
    counter = -1
    done = False
    while not done:
        try:
            pat_name = iteration[0][1] # should be the same for all, but whatever 
            sess = iteration[0][2]
            data_dict = {dt: it[-1] for (dt, it) in zip(data_types, iteration)}
            counter += 1
            yield counter, pat_name, sess, data_dict  # hope that works
        except StopIteration:
            done = True

    for pix, person in enumerate(persons):
        orig_gen = get_data(dataset, orig_datatype, False, None, split, [person], fraction_time)
        spoof_gen = get_data(dataset, spoof_datatype, spoof_name!='original', spoof_name, split, [person], fraction_time)
        data_dict = {orig_datatype: next(orig_gen)[-1], spoof_datatype: next(spoof_gen)[-1]}
        yield (pix, person, data_dict)



def get_data(dataset, data_type, spoof=False, spoof_name=None, split='all', subject_ix=None, fraction_time=None, sessions=None):
    ''' Yield 4tuple: index number, patient name, session name, ndarray with data for that patient of shape (num_leads, num_samples) 
        Can take either a split (train, test, all) or a list of patient names

        IMPORTANT: new 'fraction_time' should be a tuple (0, 0.5) for example

    '''
    if spoof_name and spoof_name != 'original' and not spoof:
        print("You provided a spoof_name but spoof=False; assuming you want it True. Come back and fix me though")
        spoof = True

    # First get the base path and list of patients, prioritizing the subject_ix list
    base_path, optional_alt_base_path = get_read_path(dataset, spoof, spoof_name)

    if not subject_ix:
        persons = get_list_of_patients(dataset, spoof, spoof_name, split)
    else:
        if not isinstance(subject_ix, list):  persons = [subject_ix]
        else:  persons = subject_ix

    data_type = data_type.lower()

    if spoof:
        # If spoofed, I generated this data and saved with a consistent naming scheme

        if spoof_name == 'their_cardiogan':
            yield from _get_data_their_cardiogan(data_type, base_path, persons, fraction_time)
        else:  # data created from my own trained cyclegan -- regardless of dataset, it's saved in a specific way
            if dataset == 'ecgfitness':
                yield from _get_data_cyclegan_ecgfitness(data_type, base_path, persons, fraction_time)
            elif dataset == 'ecgfitnessstitched':
                yield from _get_data_cyclegan_ecgfitnessstitched(data_type, base_path, persons, fraction_time)
            elif dataset == 'pure':
                yield from _get_data_cyclegan_pure(data_type, base_path, persons, fraction_time)
            elif dataset == 'hcitagging':
                yield from _get_data_cyclegan_hcitagging(data_type, base_path, persons, fraction_time)
            else:
                yield from _get_data_cyclegan(data_type, base_path, persons, fraction_time)
            
    else:
        # Not spoofed -- load original data, various ways they are saved

        if dataset == 'bidmc':
            yield from _get_data_bidmc(data_type, base_path, persons, fraction_time)
        elif dataset == 'capno':
            yield from _get_data_capno(data_type, base_path, persons, fraction_time)
        elif dataset == 'dalia':
            yield from _get_data_dalia(data_type, base_path, persons, fraction_time)
        elif dataset == 'wesad':
            yield from _get_data_wesad(data_type, base_path, persons, fraction_time)
        elif dataset == 'cardiogan4':
            # Recursively yield from a chain of the four, messy because we resample to 128
            # TODO this is only fs for ecg
            fs = [125, 300, 700, 700]
            datasetnames = ['bidmc', 'capno', 'dalia', 'wesad']
            chained = [get_data('bidmc', data_type, spoof=spoof, spoof_name=spoof_name, split=split, fraction_time=fraction_time),
                        get_data('capno', data_type, spoof=spoof, spoof_name=spoof_name, split=split, fraction_time=fraction_time),
                        get_data('dalia', data_type, spoof=spoof, spoof_name=spoof_name, split=split, fraction_time=fraction_time),
                        get_data('wesad', data_type, spoof=spoof, spoof_name=spoof_name, split=split, fraction_time=fraction_time)]
            for dname, f, c in zip(datasetnames, fs, chained):
                # Resample to 128 then yield, because once we yield it, we have no way of knowing what the orig fs for each of them was.
                for pix, patient, ecg in c:
                    resampled = processing.resample_signal(ecg, f, 128)
                    pat_name = dname+'_'+patient
                    yield pix, pat_name, resampled  # good lord why do we do this to ourselves
                
        elif dataset == 'ptt':
            yield from _get_data_ptt(data_type, base_path, persons, fraction_time=fraction_time) 
        # Below: video
        elif dataset == 'hcitagging':
            yield from _get_data_hcitagging(data_type, base_path, persons, fraction_time, sessions)
        elif dataset == 'ubfcphys':
            yield from _get_data_ubfcphys(data_type, base_path, optional_alt_base_path, persons, fraction_time, sessions)
        elif dataset == 'ecgfitness':
            yield from _get_data_ecgfitness(data_type, base_path, optional_alt_base_path, persons, fraction_time, sessions)
            """ # Deprecate all this below, now generating sessions independently 
        elif dataset == 'ecgfitnessstitched':
            yield from _get_data_ecgfitnessstitched(data_type, base_path, optional_alt_base_path, persons, fraction_time)
        elif dataset == 'ecgfitness-01': # UGH, PATCH
            yield from _get_data_ecgfitness(data_type, base_path, optional_alt_base_path, persons, fraction_time, sessions='01')
        elif dataset == 'ecgfitness-02':
            yield from _get_data_ecgfitness(data_type, base_path, optional_alt_base_path, persons, fraction_time, sessions='02')
        elif dataset == 'ecgfitness-05':
            yield from _get_data_ecgfitness(data_type, base_path, optional_alt_base_path, persons, fraction_time, sessions='05')
        elif dataset == 'ecgfitness-06':
            yield from _get_data_ecgfitness(data_type, base_path, optional_alt_base_path, persons, fraction_time, sessions='06')
            """ 
        elif dataset == 'pure':  # Video + PPG
            yield from _get_data_pure(data_type, base_path, persons, fraction_time, sessions)
        # Below: just ECG
        elif dataset == 'ecgid':
            yield from _get_data_ecgid(data_type, base_path, persons)
        elif dataset == 'ptbd':
            yield from _get_data_ptbd(data_type, base_path, persons)
        # Below BCG, SCG
        elif dataset == 'bedbased':
            yield from _get_data_bedbased(data_type, base_path, persons, fraction_time)
        elif dataset == 'cebsdb':
            yield from _get_data_cebsdb(data_type, base_path, persons, fraction_time)
        elif dataset == 'bedbased3d':
            yield from _get_data_bedbased3d(data_type, base_path, persons, fraction_time)
        elif dataset == 'bedbasedavg':
            yield from _get_data_bedbasedavg(data_type, base_path, persons, fraction_time)
        else:
            raise NotImplementedError


def _get_data_bidmc(data_type, base_path, persons, fraction_time):
    for pix, person in enumerate(persons):
        fn = os.path.join(base_path, f'bidmc_{person}_Signals.csv')
        data = pd.read_csv(fn)
        data.columns = [c.strip() for c in data.columns]
        if data_type == 'ecg':
            ecg = data['II'].to_numpy()
            if fraction_time:
                ecg = ecg[int(fraction_time[0]*ecg.size) : int(fraction_time[1]*ecg.size)]
            ecg = np.expand_dims(ecg, 0)
            yield (pix, person, SESSION_DEFAULT, ecg)
        elif data_type == 'ppg':
            ppg = data['PLETH'].to_numpy()
            if fraction_time:
                ppg = ppg[int(fraction_time[0]*ppg.size) : int(fraction_time[1]*ppg.size)]
            ppg = np.expand_dims(ppg, 0)
            yield (pix, person, SESSION_DEFAULT, ppg)
        else:
            raise NotImplementedError

def _get_data_capno(data_type, base_path, persons, fraction_time):
    for pix, person in enumerate(persons):
        s = os.path.join(base_path, person+"_8min.mat")
        data = mat73.loadmat(s)
        if data_type == 'ecg':
            ecg = data['signal']['ecg']['y']
            if fraction_time:
                ecg = ecg[int(fraction_time[0]*ecg.size) : int(fraction_time[1]*ecg.size)]
            ecg = np.expand_dims(ecg, 0)
            yield (pix, person, SESSION_DEFAULT, ecg)
        elif data_type == 'ppg':
            ppg = data['signal']['pleth']['y']
            if fraction_time:
                ppg = ppg[int(fraction_time[0]*ppg.size) : int(fraction_time[1]*ppg.size)]
            ppg = np.expand_dims(ppg, 0)
            yield (pix, person, SESSION_DEFAULT, ppg)
        else:
            raise NotImplementedError


def _get_data_dalia(data_type, base_path, persons, fraction_time):
    for pix, person in enumerate(persons):
        s = os.path.join(base_path, person)
        if data_type == 'ecg':
            ecg = np.load(os.path.join(s, 'ecg.npy')).T
            if fraction_time:
                ecg = ecg[:, int(fraction_time[0]*ecg.size) : int(fraction_time[1]*ecg.size)]
            yield (pix, person, SESSION_DEFAULT, ecg)
        elif data_type == 'ppg':
            ppg = np.load(os.path.join(s, 'ppg.npy')).T
            if fraction_time:
                ppg = ppg[:, int(fraction_time[0]*ppg.size) : int(fraction_time[1]*ppg.size)]
            yield (pix, person, SESSION_DEFAULT, ppg)
        else:
            raise NotImplementedError

def _get_data_wesad(data_type, base_path, persons, fraction_time):
    for pix, person in enumerate(persons):
        s = os.path.join(base_path, person)
        if data_type == 'ecg':
            ecg = np.load(os.path.join(s, 'ecg.npy')).T
            if fraction_time:
                ecg = ecg[:, int(fraction_time[0]*ecg.size) : int(fraction_time[1]*ecg.size)]
            yield (pix, person, SESSION_DEFAULT, ecg)
        elif data_type == 'ppg':
            ppg = np.load(os.path.join(s, 'ppg.npy')).T
            if fraction_time:
                ppg = ppg[:, int(fraction_time[0]*ppg.size) : int(fraction_time[1]*ppg.size)]
            yield (pix, person, SESSION_DEFAULT, ppg)
        else:
            raise NotImplementedError

def _get_data_ptt(data_type, base_path, persons, fraction_time):
    for pix, person in enumerate(persons):
        s = os.path.join(base_path, person+"_sit")  # just the sit ones for now
        rec = wfdb.io.rdrecord(s)
        if data_type == 'ecg':
            ecg = rec.p_signal[:,0]
            ecg = np.expand_dims(ecg, 0)  # 1 x 245902
            if fraction_time:  ecg = ecg[:, int(fraction_time[0]*ecg.size) : int(fraction_time[1]*ecg.size)]
            yield (pix, person, SESSION_DEFAULT, ecg)
        elif data_type == 'ppg':
            ppg = rec.p_signal[:,1] # there are a punch though
            ppg = np.expand_dims(ppg, 0)
            if fraction_time: ppg = ppg[:, int(fraction_time[0]*ppg.size) : int(fraction_time[1]*ppg.size)]
            yield (pix, person, SESSION_DEFAULT, ppg)
        else:
            raise ValueError

def _get_data_their_cardiogan(data_type, base_path, persons):
    assert data_type == 'ecg'
    for pix, person in enumerate(persons):
        fn = os.path.join(base_path, person, 'ecg.npy')
        ecg = np.load(fn)
        yield (pix, person, SESSION_DEFAULT, ecg)

def _get_data_my_cardiogan(data_type, base_path, persons):
    # Hardcoding, only get ecg and optionally rpeaks.
    raise DeprecationWarning  # i dont know how you would have even got here
    for pix, person in enumerate(persons):
        fn = os.path.join(base_path, person, data_type+'.npy')
        sig = np.load(fn)
        sig = np.expand_dims(sig, 0)
        yield (pix, person, SESSION_DEFAULT, sig)

def _get_data_cyclegan(data_type, base_path, persons, fraction_time):
    # Get data saved using generate_and_save.py in the main egm-ecg-cyclegan dir
    for pix, person in enumerate(persons):
        fn = os.path.join(base_path, person, data_type+'.npy')
        ecg = np.load(fn)
        if len(ecg.shape) == 1:  ecg = np.expand_dims(ecg, 0)
        if fraction_time:
            ecg = ecg[:, int(fraction_time[0]*ecg.size) : int(fraction_time[1]*ecg.size)]
        yield (pix, person, SESSION_DEFAULT, ecg)
   
def _get_data_cyclegan_ecgfitness(data_type, base_path, persons, fraction_time):
    # Get data saved using generate_and_save.py in the main egm-ecg-cyclegan dir
    for pix, person in enumerate(persons):
        for session in ['01', '02', '05', '06']:
            fn = os.path.join(base_path+session, person, data_type+'.npy')
            if not os.path.exists(fn):  continue
            ecg = np.load(fn)
            if len(ecg.shape) == 1:  ecg = np.expand_dims(ecg, 0)
            if fraction_time:
                ecg = ecg[:, int(fraction_time[0]*ecg.size) : int(fraction_time[1]*ecg.size)]
            yield (pix, person, SESSION_DEFAULT, ecg)

def _get_data_cyclegan_ecgfitnessstitched(data_type, base_path, persons, fraction_time):
    # Get data saved using generate_and_save.py in the main egm-ecg-cyclegan dir
    for pix, person in enumerate(persons):
        ECGS = []
        for session in ['01', '02', '05', '06']:
            fn = os.path.join(base_path+session, person, data_type+'.npy')
            if not os.path.exists(fn):  continue
            ecg = np.load(fn)
            if fraction_time:  ecg = ecg[int(fraction_time[0]*ecg.size) : int(fraction_time[1]*ecg.size)]
            ECGS.append(ecg)
        ecg = np.concatenate(ECGS)
        ecg = np.expand_dims(ecg, 0)
        yield (pix, person, SESSION_DEFAULT, ecg)

def _get_data_cyclegan_pure(data_type, base_path, persons, fraction_time):
    # Yield sessions
    for pix, person in enumerate(persons):
        for session in ['01', '02', '03', '04', '05', '06']:
            fn = os.path.join(base_path, person, session, data_type+'.npy')
            if not os.path.exists(fn): continue
            ppg = np.load(fn)
            if fraction_time: ppg = ppg[int(fraction_time[0]*ppg.size) : int(fraction_time[1]*ppg.size)]
            ppg = np.expand_dims(ppg, 0)
            yield(pix, person, session, ppg)

def _get_data_cyclegan_hcitagging(data_type, base_path, persons, fraction_time):
    # Yield sessions
    assert data_type == 'ecg' 
    for pix, person in enumerate(persons):
        sessions = os.listdir(os.path.join(base_path, person))
        for session in sessions: 
            if session.endswith('npy'):  continue # cheat for is not dir
            fn = os.path.join(base_path, person, session, 'ecg.npy')
            sig = np.load(fn)
            if fraction_time: sig = sig[int(fraction_time[0]*sig.size) : int(fraction_time[1]*sig.size)]
            sig = np.expand_dims(sig, 0)
            yield (pix, person, session, sig)

def _get_data_ecgfitness(data_type, base_path, optional_alt_base_path, subject_ix, fraction_time, sessions=['01', '02', '05', '06']):
    """
    Note, yielding sessions separately. 
    """
    assert data_type in ['ecg', 'rppg', 'video']
    if sessions is None:  sessions = ['01', '02', '05', '06'] # all of them 
    if not isinstance(sessions, list):  sessions = [sessions]

    # Okay before I change everything, just return the correct way for 01-1
    for pix, patient in enumerate(subject_ix):
        if data_type == 'ecg': # ECG is orig, is on Seagate

            for session in sessions:
                if session == '02' and patient == '07':  continue # TODO idk 
                full_file = os.path.join(base_path, patient, session, 'viatom-raw.csv')
                frame_file = os.path.join(base_path, patient, session, 'c920.csv')

                full_np = np.loadtxt(full_file, skiprows=1, delimiter=',')[:,1]  # 1 is ECG
                frame_ixs = np.loadtxt(frame_file, skiprows=0, delimiter=',')
                frame_ixs = frame_ixs[:,1].astype('int32')
                frame_ixs = frame_ixs[frame_ixs < full_np.shape[0]-1]
                start_ix, end_ix = frame_ixs[0], frame_ixs[-1]
                full_np = full_np[start_ix:end_ix]
                """ For now remove this -- thsi was reampling to the video 30Hz frame rate
                TODO there's still something slightly wrong here -- this starts before the 
                    video, so it's not exactly simultaneous
                frame_ixs = np.loadtxt(frame_file, skiprows=0, delimiter=',')
                frame_ixs = frame_ixs[:,1].astype('int32')
                # nb, for some reason it goes one over; confirmed with their own code
                frame_ixs = frame_ixs[frame_ixs < full_np.shape[0]-1]
                ecg = full_np[frame_ixs]
                """
                ecg = np.expand_dims(full_np, 0)
                if fraction_time: ecg = ecg[:, int(fraction_time[0]*ecg.size) : int(fraction_time[1]*ecg.size)]
                yield pix, patient, session, ecg 


        elif data_type == 'rppg':  # RPPG is generated, is at alt base path
            fn = os.path.join(optional_alt_base_path, 'PROCESSED_RPPG', patient)
            #names = ['CHROM_c920-1_0_30_rPPG.npy', 'CHROM_c920-1_30_30_rPPG.npy']  # had to do each in 2 parts
            names = ['CHROM_rppg_0_30.npy', 'CHROM_rppg_30_30.npy']
            for session in sessions:
                if session == '02' and patient == '07':  continue  # idk why this is so short
                #for v in ['1','2']:
                rppg = np.concatenate([np.load(os.path.join(fn,session,'1',n)) for n in names])
                rppg = np.expand_dims(rppg, 0)
                if fraction_time: rppg = rppg[:, int(fraction_time[0]*rppg.size) : int(fraction_time[1]*rppg.size)]
                yield  pix, patient, session, rppg

        else:
            raise NotImplementedError


def _get_data_ecgfitnessstitched(data_type, base_path, optional_alt_base_path, subject_ix, fraction_time, sessions=['01', '02', '05', '06']):
    """
    Bah, just have 2 separtate ways to do this until we figure out which is best
    """
    assert data_type in ['ecg', 'rppg']  # TODO video i guess
    if not isinstance(sessions, list):  sessions = [sessions]

    # Okay before I change everything, just return the correct way for 01-1
    for pix, patient in enumerate(subject_ix):
        if data_type == 'ecg': # ECG is orig, is on Seagate

            ecgs = []
            for session in sessions:
                if session == '02' and patient == '07':  continue # TODO idk 
                full_file = os.path.join(base_path, patient, session, 'viatom-raw.csv')
                frame_file = os.path.join(base_path, patient, session, 'c920.csv')

                full_np = np.loadtxt(full_file, skiprows=1, delimiter=',')[:,1]  # 1 is ECG
                frame_ixs = np.loadtxt(frame_file, skiprows=0, delimiter=',')
                frame_ixs = frame_ixs[:,1].astype('int32')
                frame_ixs = frame_ixs[frame_ixs < full_np.shape[0]-1]
                start_ix, end_ix = frame_ixs[0], frame_ixs[-1]
                ecg = full_np[start_ix:end_ix]
                """ For now remove this -- thsi was reampling to the video 30Hz frame rate
                TODO there's still something slightly wrong here -- this starts before the 
                    video, so it's not exactly simultaneous
                frame_ixs = np.loadtxt(frame_file, skiprows=0, delimiter=',')
                frame_ixs = frame_ixs[:,1].astype('int32')
                # nb, for some reason it goes one over; confirmed with their own code
                frame_ixs = frame_ixs[frame_ixs < full_np.shape[0]-1]
                ecg = full_np[frame_ixs]
                """
                if fraction_time: ecg = ecg[int(fraction_time[0]*ecg.size) : int(fraction_time[1]*ecg.size)]
                ecgs.append(ecg)

            ecg = np.concatenate(ecgs)
            ecg = np.expand_dims(ecg, 0)
            yield pix, patient, ecg 


        elif data_type == 'rppg':  # RPPG is generated, is at alt base path
            fn = os.path.join(optional_alt_base_path, 'PROCESSED_RPPG', patient)
            #names = ['CHROM_c920-1_0_30_rPPG.npy', 'CHROM_c920-1_30_30_rPPG.npy']  # had to do each in 2 parts
            names = ['CHROM_rppg_0_30.npy', 'CHROM_rppg_30_30.npy']
            rppgs = []
            for session in sessions:
                if session == '02' and patient == '07':  continue  # idk why this is so short
                #for v in ['1','2']:
                rppg = np.concatenate([np.load(os.path.join(fn,session,'1',n)) for n in names])
                if fraction_time: rppg = rppg[int(fraction_time[0]*rppg.size) : int(fraction_time[1]*rppg.size)]
                rppgs.append(rppg)
            rppg = np.concatenate(rppg)    
            rppg = np.expand_dims(rppg, 0)
            yield  pix, patient, rppg

        else:
            raise NotImplementedError
        

def _get_data_pure(data_type, base_path, subject_ix, fraction_time, sessions):
    """
        Yielding each mode as a separate thing 
    """
    assert data_type in ['video', 'ppg', 'rppg']

    if not sessions:
        sessions = ['01', '02', '03', '04', '05', '06']
    else:
        if not isinstance(sessions, list):
            sessions = [sessions]

    for pix, patient in enumerate(subject_ix):
        for m in sessions:
            # 06 doesn't have 02
            if patient == '06' and m == '02':  continue
            p_dirname = patient+'-'+m
            p_dir = os.path.join(base_path, p_dirname)
            if data_type == 'ppg':
                with open(os.path.join(p_dir, p_dirname+'.json'), 'r') as lfp:
                    labels = json.load(lfp)
                ppg = [label['Value']['waveform'] for label in labels['/FullPackage']]
                ppg = np.expand_dims(ppg, 0)
                if fraction_time: ppg = ppg[:, int(fraction_time[0]*ppg.size) : int(fraction_time[1]*ppg.size)]
                yield pix, patient, m, ppg
            elif data_type == 'video':
                # Just return the video filedir here 
                yield pix, patient, m, os.path.join(base_path, p_dirname, p_dirname)
            elif data_type == 'rppg':
                fn = os.path.join(base_path, 'PROCESSED_RPPG', patient, m, 'CHROM_rppg_0_60.npy')
                rppg = np.load(fn)
                rppg = np.expand_dims(rppg, 0)
                if fraction_time: rppg = rppg[:, int(fraction_time[0]*rppg.size) : int(fraction_time[1]*rppg.size)]
                yield pix, patient, m, rppg
            else:
                raise NotImplementedError
                
        

    """
    for pix, patient in enumerate(subject_ix):
        fn = os.path.join(base_path, patient, '03', 'viatom-raw.csv') # 3 is halogen rowing, 4 is halogen speaking
        df = pd.read_csv(fn)
        df.columns = [c.strip() for c in df.columns]
        if data_type == 'ecg':
            for m in modes:  # Return these separately 
                raise NotImplementedError # TODO TODO COME BACK HERE, DECIDE WHAT TO CONCAT AND WHAT TO NOT
            sig = np.expand_dims(df['ECG'], 0)
            if fraction_time:  sig = sig[:, int(fraction_time[0]*sig.size) : int(fraction_time[1]*sig.size)]

        elif data_type == 'rppg':
            rppgs = []
            for ty in modes: 
                rppg_fns = glob.glob(os.path.join(optional_alt_base_path, 'PROCESSED_RPPG', patient, ty, '1', f'CHROM_*.npy'))  # TODO only giving us video 1 for now
                rppg_fns.sort(key = lambda x: int(x.split('/')[-1].split('_')[-2]))
                rppgs.extend([np.load(f) for f in rppg_fns])
            sig = np.concatenate(rppgs, 0)
            sig = np.expand_dims(sig, 0)
            if fraction_time: sig = sig[:, int(fraction_time[0]*sig.size) : int(fraction_time[1]*sig.size)]

        elif data_type == 'video':
            # just the first one for now..
            sig = os.path.join(base_path, patient, '03', 'c920-1.avi')
        yield (pix, patient, sig)
    """


def _get_data_ubfcphys(data_type, base_path, optional_alt_base_path, subject_ix, fraction_time, sessions):
    """
        TODO - not sure this is the right way to do this, but for now i'm yielding 3 times per subject
        So the len of the generator is 3x the number of patients
        To handle the T1, T2, T3 without just concatenating them and having weird edge mismatches
    """
    assert data_type in ['video', 'ppg', 'rppg']
    assert not (data_type == 'video' and fraction_time), "Youre doing something wrong"

    if not sessions: sessions = ['T1', 'T2', 'T3']
    if not isinstance(sessions, list):  sessions = [sessions]

    for pix, patient in enumerate(subject_ix):

        # TODO this is just a patch for residual ones, coudlnt fit all on Seagate 
        if not os.path.exists(os.path.join(base_path, patient)):
            base_path = "/media/big_hdd/kveena/pyvhr/UBFC-PHYS"

        # Get one avi
        if data_type == 'video':  # BEWARE THIS IS NEVER REALLY USED
            for ty in sessions:
                avi = os.path.join(base_path, patient, f'vid_{patient}_{ty}.avi')
                yield pix, patient, ty, avi
        elif data_type == 'ppg':
            for session in sessions: 
                ppg = np.loadtxt(os.path.join(base_path, patient, f'bvp_{patient}_{session}.csv'))
                ppg = np.expand_dims(ppg, 0)
                if fraction_time: ppg = ppg[:, int(fraction_time[0]*ppg.size) : int(fraction_time[1]*ppg.size)]
                yield pix, patient, session, ppg
        elif data_type == 'rppg':
            for session in sessions: 
                rppg_fns = glob.glob(os.path.join(optional_alt_base_path, 'PROCESSED_RPPG', patient, session, f'CHROM_*.npy'))
                rppg_fns.sort(key = lambda x: int(x.split('/')[-1].split('_')[-2]))
                rppgs = [np.load(f) for f in rppg_fns]
                if not rppgs:  continue  # some sessions didnt work out
                rppg = np.concatenate(rppgs, 0)
                rppg = np.expand_dims(rppg, 0)
                if fraction_time: rppg = rppg[:, int(fraction_time[0]*rppg.size) : int(fraction_time[1]*rppg.size)]
                yield pix, patient, session, rppg

 
def _get_data_hcitagging(data_type, base_path, subject_ix, fraction_time, sessions):
    assert data_type in ['video', 'ecg', 'rppg', 'ppg'] # the ppg is synthesized
    with open(os.path.join(base_path, 'subject_session_mapping.pkl'), 'rb') as fp:
        session_mapping = pkl.load(fp)

    get_all_sessions = not sessions  # flag for which sessions to enumerate later
    if not isinstance(sessions, list):  sessions = [sessions]

    for pix, patient in enumerate(subject_ix):

        if get_all_sessions:  sessions = session_mapping[patient]  # otherwise only supplied

        if data_type == 'video':
            for sess in sessions:
                avi_file = glob.glob(os.path.join(base_path, 'Sessions', sess, '*.avi'))[0]
                yield (pix, patient, sess, avi_file)
        elif data_type == 'ecg':
            # Read the bdf file and get the ecg
            # six cameras, C1 (color, frontal view, prob best), 2, 3, 4, 5 (side view), 6 (far away)
            for sess in sessions:
                bdffile = glob.glob(os.path.join(base_path, 'Sessions', sess, '*.bdf'))
                # ECG are channels 33 (EXG1), 34 (EXG2), and 35 (EXG3)  -- get just lead1 for now 
                if not bdffile:  import pdb; pdb.set_trace()
                ecg, _, _ = highlevel.read_edf(bdffile[0], ch_names=['EXG1'])
                ecg = ecg[:, 256*30: -256*30]  # extra 30 secs before/after
                if fraction_time:  ecg = ecg[:, int(fraction_time[0]*ecg.size) : int(fraction_time[1]*ecg.size)]
                yield (pix, patient, sess, ecg)  # TODO BEWARE 
        elif data_type == 'rppg':
            # Stitch together multiple CHROMs for session 
            for sess in sessions:
                bdir = os.path.join(base_path, 'PROCESSED_RPPG', patient, sess)
                choms = os.listdir(bdir)
                if not choms:  print("ERROR: NO CHROMS FOR ", patient, sess)
                choms.sort(key = lambda x: int(x.split('/')[-1].split('_')[-2]))
                rppgs = [np.load(os.path.join(bdir, f)) for f in choms]
                rppg = np.concatenate(rppgs, 0)
                rppg = np.expand_dims(rppg, 0)
                if fraction_time: rppg = rppg[:, int(fraction_time[0]*rppg.size) : int(fraction_time[1]*rppg.size)]
                yield pix, patient, sess, rppg
        elif data_type == 'ppg':
            p = '/media/big_hdd/kveena/spoofed/combined/rppgecg_hcitagging'
            for sess in sessions:
                ppg = np.load(os.path.join(p, patient, sess, 'ecg.npy')) # saved as ecg cause of how generate_save works 
                ppg = np.expand_dims(ppg, 0)
                if fraction_time: ppg = ppg[:, int(fraction_time[0]*ppg.size) : int(fraction_time[1]*ppg.size)]
                yield pix, patient, sess, ppg
                
                
        else:
            raise NotImplementedError


def _get_data_ecgid(data_type, base_path, subject_ix):
    assert data_type == 'ecg'
    for pix, person in enumerate(subject_ix):
        ecg = []
        # We have a different number of ecg recordings per person. Sometimes 1, sometimes 20. 
        # Just stack them anyway and deal with it later
        rec_list = [os.path.splitext(x)[0] for x in glob.glob(os.path.join(base_path, person, '*.hea'))]
        # TODO hack, just get the first for now
        rec_list = [rec_list[0]]
        for rec_name in rec_list:
            rec = wfdb.io.rdrecord(rec_name)
            filtered = rec.p_signal[:,1]
            ecg.append(filtered)
        ecg = np.stack(ecg)
        yield(pix, person, SESSION_DEFAULT, ecg)  

def _get_data_cybhi(data_type, base_path, persons):
    assert data_type == 'ecg'
    for pix, person in enumerate(persons):
        fn = os.path.join(base_path, persons+".txt")
        with open(fn) as fp:
            for _ in range(6):  next(fp)
            lines = [int(line.strip()) for line in fp] 
        # Now normalize
        line_min = min(lines)
        line_max = max(lines)
        line_spread = line_max - line_min
        ecg = [ (l - min_min)/line_spread for l in lines ]
        ecg = np.array(ecg).expand_dims(0)
        yield (pix, person, SESSION_DEFAULT, ecg)


def _get_data_ptbd(data_type, base_path, persons):
    assert data_type == 'ecg'
    for pix, person in enumerate(persons):
        rec_list = [os.path.splitext(x)[0] for x in glob.glob(os.path.join(base_path, person, '*.hea'))]
        rec_name = rec_list[0] # HACK, deal with multiple sessions later
        rec = wfdb.io.rdrecord(rec_name)
        ecg = rec.p_signal # shape is (115200, 15)
        ecg = ecg.T  # NOTE 15xlen
        yield (pix, person, SESSION_DEFAULT, ecg)


def _get_data_bedbased(data_type, base_path, persons, fraction_time):
    for pix, person in enumerate(persons):
        data = pd.read_csv(os.path.join(base_path, person + ".csv"))
        if data_type == 'ecg':
            key = 'ECG'
        elif data_type == 'bcg':
            key = 'LC_BCG1'
        elif data_type == 'ppg':
            key = 'PPG'
        else:
            raise ValueError

        sig = np.expand_dims(data[key], 0)  # 1xn ndarray
        if fraction_time:  sig = sig[:, int(fraction_time[0]*sig.size) : int(fraction_time[1]*sig.size)]
        yield (pix, person, SESSION_DEFAULT, sig)
        
def _get_data_bedbased3d(data_type, base_path, persons, fraction_time):
    for pix, person in enumerate(persons):
        data = pd.read_csv(os.path.join(base_path, person + ".csv"))
        # Channels are ['PPG', 'Resp', 'ECG', 'Film0', 'Film1', 'Film2', 'Film3', 'LC_BCG0', 'LC_BCG1', 'LC_BCG2', 'LC_BCG3', 'reBAP', 'IBI', 'SV', 'dp_dt']
        if data_type == 'ecg':
            key = 'ECG'
        elif data_type == 'bcg':
            key = ['LC_BCG0', 'LC_BCG1', 'LC_BCG2', 'LC_BCG3']  # load cells => average? 
        elif data_type == 'ppg':
            key = 'PPG'
        else:
            raise ValueError

        sig = np.expand_dims(data[key], 0)  # 1xn ndarray
        if fraction_time:  sig = sig[:, int(fraction_time[0]*sig.size) : int(fraction_time[1]*sig.size)]
        yield (pix, person, SESSION_DEFAULT, sig)

def _get_data_bedbasedavg(data_type, base_path, persons, fraction_time):
    for pix, person in enumerate(persons):
        data = pd.read_csv(os.path.join(base_path, person + ".csv"))
        # Channels are ['PPG', 'Resp', 'ECG', 'Film0', 'Film1', 'Film2', 'Film3', 'LC_BCG0', 'LC_BCG1', 'LC_BCG2', 'LC_BCG3', 'reBAP', 'IBI', 'SV', 'dp_dt']
        if data_type == 'ecg':
            key = 'ECG'
            sig = np.expand_dims(data[key], 0)  # 1xn ndarray
        elif data_type == 'bcg':
            key = ['LC_BCG0', 'LC_BCG1', 'LC_BCG2', 'LC_BCG3']  # load cells => average? 
            sig = data[key].mean(1)
            sig = np.expand_dims(sig, 0)
        elif data_type == 'ppg':
            key = 'PPG'
            sig = np.expand_dims(data[key], 0)  # 1xn ndarray
        else:
            raise ValueError

        if fraction_time:  sig = sig[:, int(fraction_time[0]*sig.size) : int(fraction_time[1]*sig.size)]
        yield (pix, person, SESSION_DEFAULT, sig)

def _get_data_cebsdb(data_type, base_path, persons, fraction_time):
    for pix, person in enumerate(persons):
        rec_name = os.path.join(base_path, person)
        rec = wfdb.io.rdsamp(rec_name)
        data = rec[0]  # shape is 1360K, 4. fields are ['I', 'II', 'RESP', 'SCG']
        if data_type == 'ecg':
            sig = data[:,0] 
        elif data_type == 'scg':
            sig = data[:,3]
        elif data_type == 'rr':
            sig = data[:, 2]
        else:
            raise NotImplementedError
        sig = np.expand_dims(sig, 0)
        if fraction_time:  sig = sig[:, int(fraction_time[0]*sig.size) : int(fraction_time[1]*sig.size)]
        yield (pix, person, SESSION_DEFAULT, sig)


