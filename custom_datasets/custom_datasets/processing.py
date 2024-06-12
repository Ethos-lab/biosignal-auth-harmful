# Simple things like getting ground truth rpeaks and etc
import os
import numpy as np
import pandas as pd
from custom_datasets.properties import get_props
import scipy.stats as scipy_stats
import scipy.signal as scipy_signal
import neurokit2 as nk2
from biosppy.signals import tools as tools
from math import floor


def resample_get_rpeaks(ecg, orig_fs, resample_to=128):
    # first get true fs from props and base path 
    resampled = resample_signal(ecg, orig_fs, resample_to)
    rpeaks = get_ecg_rpeaks_hamilton(resampled, resample_to)
    return resampled, rpeaks

def resample_signal(signal, orig_fs, target_fs):  
    num_samp_orig = signal.shape[1]
    num_samp_target = round( (target_fs * num_samp_orig) / orig_fs )
    if num_samp_orig == num_samp_target:  return signal
    resized = scipy_signal.resample(signal, num_samp_target, axis=1)
    return resized

def resample_rpeak_arr(rpeaks, orig_fs, target_fs):
    # basically a reindexing/mapping problem
    scaling = target_fs / orig_fs
    resampled_rpeaks = [floor(r*scaling) for r in rpeaks]
    return resampled_rpeaks

def get_ecg_rpeaks_hamilton(ecg, fs):
    cleaned = nk2.ecg_clean(ecg[0], sampling_rate=fs, method='hamilton2002')
    _, res = nk2.ecg_peaks(cleaned, sampling_rate=fs, method='hamilton2002')
    peaks = res['ECG_R_Peaks']
    return peaks

def get_rlocks_adjusted_ptt(source_type, source_sig, pred_bcg, fs):
    # Get rlocs in ecg
    # But then shift them to account for the shift that we got when we generated the target scg
    # Specifically for Zhang BCG right now 
    if source_type == 'ecg':
        # Get rlocs from ecg, there's no shift 
        return get_ecg_rpeaks_hamilton(source_sig, fs)
    elif source_type == 'ppg':
        raise NotImplementedError

def min_max_norm(sig, r=[-1, 1]):
    ' From cardiogan paper: [-1, 1] normalize each windowed segment. Done per lead '
    nl = sig.shape[0]
    for l in range(nl):
        minx = sig[l].min()
        maxx = sig[l].max()
        newl = r[0] + ( (sig[l] - minx)*(r[1] - r[0]) ) / (maxx - minx)
        sig[l] = newl
    return sig

def zscore(sig):
    mu, std = np.mean(sig), np.std(sig)
    sig = sig - mu
    sig = sig / std
    return sig
normalize = zscore  # just because i changed the name of this at some point...

def filter_signal(signal, freqs, fs, order=3):

    signal = np.array(signal)
    sampling_rate = float(fs)
    filtered, _, _ = tools.filter_signal(signal=signal,
                                  ftype='butter',
                                  band='bandpass',  # going by Ballistocardiogram ... paper
                                  order=order,
                                  frequency=freqs,
                                  sampling_rate=sampling_rate)
    return filtered

def sliding_window(data, fs, window_len=4, overlap_len=0.4): # window_len and overlap_len in seconds
    ' len should be in seconds '
    arr = []
    i = 0
    l = fs*window_len  # target num samples per window
    nl = data.shape[0]  # num leads
    ns = data.shape[1]  # num samples
    shift_len = floor((window_len - overlap_len) * fs)  # if 10% overlap, we shift 90%, or (window_len - overlap_len) seconds, times fs gives num samples to shift
    while i+l < ns:
        segment = data[:, i:i+l] # section out the window
        arr.append(segment)
        i = i + shift_len # increase the shift
    return np.stack(arr)


def resample(signal, orig_fs = 700, target_fs=128):
    ''' Reamples using scipy.signal.resample, which uses fourier interpolation. the cv2.INTER_LINEAR wasnt working for some reason.
        Make sure input signal is shape num_leads x num_sample '''
    #segment_size = int(target_fs*target_len_min*60)
    #resized = cv2.resize(signal, (1, segment_size), interpolation=cv2.INTER_LINEAR)  # this is how their code does it
    num_samp_orig = signal.shape[1]
    num_samp_target = round( (target_fs * num_samp_orig) / orig_fs )
    resized = scipy_signal.resample(signal, num_samp_target, axis=1)
    return resized


def filter_ecg(signal, sampling_rate):

    signal = np.array(signal)
    order = int(0.3 * sampling_rate)
    filtered, _, _ = tools.filter_signal(signal=signal,
                                  ftype='FIR',
                                  band='bandpass',
                                  order=order,
                                  frequency=[3, 45],
                                  sampling_rate=sampling_rate)
    return filtered


def filter_ppg(signal, sampling_rate):

    signal = np.array(signal)
    sampling_rate = float(sampling_rate)
    filtered, _, _ = tools.filter_signal(signal=signal,
                                  ftype='butter',
                                  band='bandpass',
                                  order=4, #3
                                  frequency=[1, 8], #[0.5, 8]
                                  sampling_rate=sampling_rate)
    return filtered
